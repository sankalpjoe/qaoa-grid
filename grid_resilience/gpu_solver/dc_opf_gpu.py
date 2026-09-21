"""
GPU-Accelerated Continuous DC-OPF Relaxation Solver (Target: NVIDIA RTX 5070).
Solves the continuous relaxation of DC Optimal Transmission Switching (DC-OTS)
using PyTorch with CUDA acceleration.
Maps continuous line assignments s* in [0, 1] to Bloch sphere rotation angles theta
with a regularized safe-exploration band epsilon.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..grid.network import GridNetwork


@dataclass
class GpuRelaxationResult:
    continuous_statuses: Dict[int, float]  # line_id -> s_l in [0, 1]
    bloch_angles: Dict[int, float]        # line_id -> theta_l in [0, pi]
    loss_history: List[float]
    solver_time_ms: float
    device_used: str
    thermal_violation_mw: float
    max_loading_pct: float


class ContinuousDcOpfGpu:
    """
    Solves continuous DC-OTS on CUDA GPU (RTX 5070) via differentiable power flow optimization.
    """

    def __init__(
        self,
        network: GridNetwork,
        initiating_outages: List[int],
        candidate_lines: Optional[List[int]] = None,
        epsilon_safe_band: float = 0.08,
        device: Optional[str] = None
    ):
        self.network = network
        self.initiating_outages = initiating_outages
        self.epsilon = epsilon_safe_band

        # Device selection: prefer CUDA (RTX 5070)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Candidates for switching (exclude tripped lines and radial branches)
        if candidate_lines is None:
            # All lines except tripped lines
            self.candidate_lines = [lid for lid in network.branches if lid not in initiating_outages]
        else:
            self.candidate_lines = [lid for lid in candidate_lines if lid not in initiating_outages]

        self.num_candidates = len(self.candidate_lines)
        self.candidate_idx_map = {lid: i for i, lid in enumerate(self.candidate_lines)}

    def solve(
        self,
        num_epochs: int = 250,
        learning_rate: float = 0.05,
        penalty_overload: float = 1000.0,
        penalty_switch: float = 5.0,
        penalty_slack: float = 50.0
    ) -> GpuRelaxationResult:
        """
        Runs GPU optimization using PyTorch tensors.
        """
        start_time = time.perf_counter()

        n_buses = self.network.num_buses
        n_branches = self.network.num_branches
        bus_map = {bid: i for i, bid in enumerate(sorted(self.network.buses.keys()))}
        slack_idx = bus_map[self.network.slack_bus_id]

        # Convert network parameters to tensors on GPU
        from_idx = torch.tensor([bus_map[self.network.branches[lid].from_bus] for lid in range(n_branches)], device=self.device, dtype=torch.long)
        to_idx = torch.tensor([bus_map[self.network.branches[lid].to_bus] for lid in range(n_branches)], device=self.device, dtype=torch.long)
        reactances = torch.tensor([self.network.branches[lid].x for lid in range(n_branches)], device=self.device, dtype=torch.float32)
        ratings = torch.tensor([self.network.branches[lid].rating for lid in range(n_branches)], device=self.device, dtype=torch.float32)
        base_mva = float(self.network.base_mva)

        loads = torch.tensor([self.network.buses[bid].p_load for bid in sorted(self.network.buses.keys())], device=self.device, dtype=torch.float32)
        p_gen_nom = torch.tensor([self.network.buses[bid].p_gen for bid in sorted(self.network.buses.keys())], device=self.device, dtype=torch.float32)

        # Mask for fixed non-candidate lines
        is_candidate = torch.zeros(n_branches, dtype=torch.bool, device=self.device)
        for lid in self.candidate_lines:
            is_candidate[lid] = True

        is_tripped = torch.zeros(n_branches, dtype=torch.bool, device=self.device)
        for lid in self.initiating_outages:
            is_tripped[lid] = True

        # Trainable parameters: logits for candidate switching status
        # Initialized to high logits so s ~ 1.0 (initially all lines closed)
        w_status = nn.Parameter(torch.full((self.num_candidates,), 3.5, device=self.device, dtype=torch.float32))

        # Bus angle approximations as trainable state
        bus_angles = nn.Parameter(torch.zeros(n_buses, device=self.device, dtype=torch.float32))

        optimizer = optim.Adam([w_status, bus_angles], lr=learning_rate)
        loss_history = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Continuous status s in [0, 1]
            s_cand = torch.sigmoid(w_status)

            # Full line status vector
            s_full = torch.ones(n_branches, device=self.device, dtype=torch.float32)
            s_full[is_tripped] = 0.0

            cand_indices = torch.tensor([lid for lid in self.candidate_lines], device=self.device, dtype=torch.long)
            s_full[cand_indices] = s_cand

            # Power flows: P_l = s_l * (theta_from - theta_to) / x_l * base_mva
            delta_theta = bus_angles[from_idx] - bus_angles[to_idx]
            flows = s_full * (delta_theta / reactances) * base_mva

            # Nodal balance calculation: P_inj_calc = sum(flows_out) - sum(flows_in)
            p_balance = torch.zeros(n_buses, device=self.device, dtype=torch.float32)
            p_balance.index_add_(0, from_idx, flows)
            p_balance.index_add_(0, to_idx, -flows)

            # Scheduled net injection (slack bus excluded from balance constraint)
            p_inj_sched = p_gen_nom - loads

            # Non-slack nodal balance violation
            balance_error = p_balance - p_inj_sched
            balance_loss = torch.sum(balance_error**2)

            # Thermal overloads: max(0, |flows| - rating)
            abs_flows = torch.abs(flows)
            overload = torch.relu(abs_flows - ratings)
            overload_loss = torch.sum(overload**2)

            # Switching penalty: penalize switching lines off (s < 1.0)
            switch_loss = torch.sum(1.0 - s_cand)

            # Slack bus angle fixed to 0
            slack_penalty = bus_angles[slack_idx]**2

            # Total loss
            loss = (
                penalty_overload * overload_loss
                + 10.0 * balance_loss
                + penalty_switch * switch_loss
                + penalty_slack * slack_penalty
            )

            loss.backward()
            optimizer.step()

            # Slack reference enforce
            with torch.no_grad():
                bus_angles[slack_idx] = 0.0

            loss_history.append(float(loss.item()))

        # Extract optimal continuous values
        with torch.no_grad():
            s_cand_opt = torch.sigmoid(w_status).detach().cpu().numpy()
            final_flows = flows.detach().cpu().numpy()
            final_abs_flows = np.abs(final_flows)
            final_ratings = ratings.detach().cpu().numpy()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        continuous_statuses = {}
        bloch_angles = {}

        for i, lid in enumerate(self.candidate_lines):
            s_val = float(s_cand_opt[i])
            continuous_statuses[lid] = s_val

            # Safe-exploration regularized band: clip to [epsilon, 1 - epsilon]
            s_reg = float(np.clip(s_val, self.epsilon, 1.0 - self.epsilon))
            # Bloch sphere mapping: theta = 2 * arcsin(sqrt(s_reg))
            # When s_reg close to 1 -> theta close to pi (state |1>, line closed)
            # When s_reg close to 0 -> theta close to 0 (state |0>, line switched)
            theta_val = float(2.0 * np.arcsin(np.sqrt(s_reg)))
            bloch_angles[lid] = theta_val

        # Check final metrics
        thermal_mw = float(np.sum(np.maximum(0.0, final_abs_flows - final_ratings)))
        max_load = float(np.max(final_abs_flows / np.maximum(final_ratings, 1e-6)))

        device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"

        return GpuRelaxationResult(
            continuous_statuses=continuous_statuses,
            bloch_angles=bloch_angles,
            loss_history=loss_history,
            solver_time_ms=elapsed_ms,
            device_used=f"{self.device.type} ({device_name})",
            thermal_violation_mw=thermal_mw,
            max_loading_pct=max_load
        )
