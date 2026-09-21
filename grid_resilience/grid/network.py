"""
Power Grid Network Model and Physics Formulation.
Supports IEEE 14-bus and IEEE 30-bus test systems.
Computes Bus Admittance Matrix (B), PTDF (Power Transfer Distribution Factors),
and solves DC Power Flow for given generation, load, and switching topologies.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx


@dataclass
class Branch:
    id: int
    from_bus: int
    to_bus: int
    x: float          # Reactance in p.u.
    rating: float     # Thermal rating in MW (p.u. base 100 MVA)
    in_service: bool = True


@dataclass
class Bus:
    id: int
    bus_type: str     # 'slack', 'gen', 'load'
    p_load: float     # Active load in MW
    p_gen_max: float  # Max generation in MW
    p_gen_min: float  # Min generation in MW
    p_gen: float      # Scheduled generation in MW


class GridNetwork:
    """
    Encapsulates AC transmission topology converted to linearized DC power flow.
    """

    def __init__(self, name: str, base_mva: float = 100.0):
        self.name = name
        self.base_mva = base_mva
        self.buses: Dict[int, Bus] = {}
        self.branches: Dict[int, Branch] = {}
        self.slack_bus_id: int = 1

    @classmethod
    def create_ieee14(cls, rating_multiplier: float = 1.20) -> "GridNetwork":
        """
        Creates standard IEEE 14-bus test system.
        Base MVA: 100 MVA.
        14 buses, 20 branches, 5 generation buses (1, 2, 3, 6, 8).
        """
        net = cls("IEEE-14", base_mva=100.0)
        net.slack_bus_id = 1

        # Buses (id, type, p_load, p_gen_max, p_gen_min, p_gen)
        # Values in MW
        bus_data = [
            (1, "slack", 0.0, 332.4, 0.0, 232.4),
            (2, "gen", 21.7, 140.0, 0.0, 40.0),
            (3, "gen", 94.2, 100.0, 0.0, 0.0),
            (4, "load", 47.8, 0.0, 0.0, 0.0),
            (5, "load", 7.6, 0.0, 0.0, 0.0),
            (6, "gen", 11.2, 100.0, 0.0, 0.0),
            (7, "load", 0.0, 0.0, 0.0, 0.0),
            (8, "gen", 0.0, 100.0, 0.0, 0.0),
            (9, "load", 29.5, 0.0, 0.0, 0.0),
            (10, "load", 9.0, 0.0, 0.0, 0.0),
            (11, "load", 3.5, 0.0, 0.0, 0.0),
            (12, "load", 6.1, 0.0, 0.0, 0.0),
            (13, "load", 13.5, 0.0, 0.0, 0.0),
            (14, "load", 14.9, 0.0, 0.0, 0.0),
        ]
        for bid, btype, pload, pgmax, pgmin, pgen in bus_data:
            net.buses[bid] = Bus(bid, btype, pload, pgmax, pgmin, pgen)

        # Branches (id, from_bus, to_bus, x (p.u.), rating in MW)
        # Stressed ratings to represent realistic contingency and cascading trigger points
        branch_data = [
            (0, 1, 2, 0.05917, 160.0),
            (1, 1, 5, 0.22304, 75.0),
            (2, 2, 3, 0.19797, 80.0),
            (3, 2, 4, 0.17632, 70.0),
            (4, 2, 5, 0.17388, 55.0),
            (5, 3, 4, 0.17103, 50.0),
            (6, 4, 5, 0.04211, 80.0),
            (7, 4, 7, 0.20912, 45.0),
            (8, 4, 9, 0.55618, 40.0),
            (9, 5, 6, 0.25202, 60.0),
            (10, 6, 11, 0.19890, 25.0),
            (11, 6, 12, 0.25581, 20.0),
            (12, 6, 13, 0.13027, 30.0),
            (13, 7, 8, 0.17615, 35.0),
            (14, 7, 9, 0.11001, 35.0),
            (15, 9, 10, 0.08450, 20.0),
            (16, 9, 14, 0.27038, 25.0),
            (17, 10, 11, 0.19207, 15.0),
            (18, 12, 13, 0.19988, 15.0),
            (19, 13, 14, 0.34802, 15.0),
        ]
        for lid, fb, tb, x, rate in branch_data:
            net.branches[lid] = Branch(lid, fb, tb, x, rate * rating_multiplier, in_service=True)

        return net

    @classmethod
    def create_ieee30(cls, rating_multiplier: float = 1.20) -> "GridNetwork":
        """
        Creates standard IEEE 30-bus test system (30 buses, 41 branches, 6 generators).
        """
        net = cls("IEEE-30", base_mva=100.0)
        net.slack_bus_id = 1

        # 30 buses
        bus_loads = {
            1: 0.0, 2: 21.7, 3: 2.4, 4: 7.6, 5: 94.2, 6: 0.0, 7: 22.8, 8: 30.0,
            9: 0.0, 10: 5.8, 11: 0.0, 12: 11.2, 13: 0.0, 14: 6.2, 15: 8.2, 16: 3.5,
            17: 9.0, 18: 3.2, 19: 9.5, 20: 2.2, 21: 17.5, 22: 0.0, 23: 3.2, 24: 8.7,
            25: 0.0, 26: 3.5, 27: 0.0, 28: 0.0, 29: 2.4, 30: 10.6
        }
        gen_buses = {1: (260.0, 0.0, 138.5), 2: (100.0, 0.0, 57.5), 5: (100.0, 0.0, 24.5),
                     8: (100.0, 0.0, 35.0), 11: (100.0, 0.0, 17.9), 13: (100.0, 0.0, 16.9)}

        for bid in range(1, 31):
            btype = "slack" if bid == 1 else ("gen" if bid in gen_buses else "load")
            pload = bus_loads.get(bid, 0.0)
            pgmax, pgmin, pgen = gen_buses.get(bid, (0.0, 0.0, 0.0))
            net.buses[bid] = Bus(bid, btype, pload, pgmax, pgmin, pgen)

        # 41 branches
        branches_raw = [
            (1, 2, 0.0575, 130), (1, 3, 0.1652, 130), (2, 4, 0.1737, 65), (3, 4, 0.0379, 130),
            (2, 5, 0.1983, 130), (2, 6, 0.1763, 65), (4, 6, 0.0414, 90), (5, 7, 0.1160, 70),
            (6, 7, 0.0820, 130), (6, 8, 0.0420, 32), (6, 9, 0.2080, 65), (6, 10, 0.5560, 32),
            (9, 11, 0.2080, 65), (9, 10, 0.1100, 65), (4, 12, 0.2560, 65), (12, 13, 0.1400, 65),
            (12, 14, 0.2559, 32), (12, 15, 0.1304, 32), (12, 16, 0.1987, 32), (14, 15, 0.1997, 16),
            (16, 17, 0.1923, 16), (15, 18, 0.2185, 16), (18, 19, 0.1292, 16), (19, 20, 0.0680, 32),
            (10, 20, 0.2090, 32), (10, 17, 0.0844, 32), (10, 21, 0.0749, 32), (10, 22, 0.1499, 32),
            (21, 22, 0.0236, 32), (15, 23, 0.2020, 16), (22, 24, 0.1790, 16), (23, 24, 0.2700, 16),
            (24, 25, 0.3292, 16), (25, 26, 0.3800, 16), (25, 27, 0.2087, 16), (28, 27, 0.3960, 65),
            (27, 29, 0.4153, 16), (27, 30, 0.6027, 16), (29, 30, 0.4533, 16), (8, 28, 0.2000, 32),
            (6, 28, 0.0599, 32)
        ]
        for lid, (fb, tb, x, rate) in enumerate(branches_raw):
            net.branches[lid] = Branch(lid, fb, tb, x, float(rate) * rating_multiplier, in_service=True)

        return net

    @property
    def num_buses(self) -> int:
        return len(self.buses)

    @property
    def num_branches(self) -> int:
        return len(self.branches)

    def to_networkx(self, active_only: bool = True) -> nx.Graph:
        """Constructs NetworkX graph for topology analysis and connectivity check."""
        g = nx.Graph()
        for bid in self.buses:
            g.add_node(bid)
        for lid, br in self.branches.items():
            if not active_only or br.in_service:
                g.add_edge(br.from_bus, br.to_bus, branch_id=lid, reactance=br.x, rating=br.rating)
        return g

    def is_connected(self, active_only: bool = True) -> bool:
        g = self.to_networkx(active_only=active_only)
        return nx.is_connected(g)

    def compute_b_bus(self, active_only: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds Bus Admittance Matrix B_bus and Branch Admittance Matrix B_branch.
        Returns:
            B_bus: (N, N) matrix
            B_branch: (M, M) diagonal susceptance matrix
            A: (M, N) incidence matrix
        """
        n = self.num_buses
        m = self.num_branches
        bus_idx_map = {bid: i for i, bid in enumerate(sorted(self.buses.keys()))}

        A = np.zeros((m, n), dtype=np.float64)
        b_series = np.zeros(m, dtype=np.float64)

        for lid, br in self.branches.items():
            if not active_only or br.in_service:
                i = bus_idx_map[br.from_bus]
                j = bus_idx_map[br.to_bus]
                A[lid, i] = 1.0
                A[lid, j] = -1.0
                b_series[lid] = 1.0 / br.x

        B_branch = np.diag(b_series)
        B_bus = A.T @ B_branch @ A
        return B_bus, B_branch, A

    def compute_ptdf(self, active_only: bool = True) -> np.ndarray:
        """
        Computes the Power Transfer Distribution Factor (PTDF) matrix:
        PTDF = B_branch @ A @ [B_bus_reduced]^-1
        Maps nodal net power injections (P_inj) to branch power flows (P_flow).
        Shape: (M, N)
        """
        n = self.num_buses
        m = self.num_branches
        B_bus, B_branch, A = self.compute_b_bus(active_only=active_only)

        ref_idx = self.slack_bus_id - 1
        non_slack_idx = [i for i in range(n) if i != ref_idx]

        # Invert reduced B_bus
        B_reduced = B_bus[np.ix_(non_slack_idx, non_slack_idx)]
        B_inv_red = np.linalg.pinv(B_reduced)

        B_inv_full = np.zeros((n, n), dtype=np.float64)
        for r_i, i in enumerate(non_slack_idx):
            for r_j, j in enumerate(non_slack_idx):
                B_inv_full[i, j] = B_inv_red[r_i, r_j]

        # PTDF = B_branch @ A @ B_inv_full
        ptdf = B_branch @ A @ B_inv_full
        return ptdf

    def solve_dc_power_flow(
        self,
        p_gen_override: Optional[Dict[int, float]] = None,
        line_status: Optional[Dict[int, bool]] = None
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Solves DC Power Flow.
        Returns:
            flows: (M,) array of branch power flows in MW
            angles: (N,) array of bus voltage angles in radians
            is_valid: True if grid is connected and solvable
        """
        # Save original branch status if line_status provided
        orig_status = {lid: br.in_service for lid, br in self.branches.items()}
        if line_status is not None:
            for lid, status in line_status.items():
                self.branches[lid].in_service = status

        if not self.is_connected(active_only=True):
            # Restore and return invalid
            for lid, status in orig_status.items():
                self.branches[lid].in_service = status
            return np.zeros(self.num_branches), np.zeros(self.num_buses), False

        n = self.num_buses
        m = self.num_branches
        bus_idx_map = {bid: i for i, bid in enumerate(sorted(self.buses.keys()))}

        # Net injections: P_inj = P_gen - P_load
        p_inj = np.zeros(n, dtype=np.float64)
        total_load = 0.0
        total_gen_nonslack = 0.0

        for bid, bus in self.buses.items():
            idx = bus_idx_map[bid]
            p_load = bus.p_load
            p_gen = p_gen_override.get(bid, bus.p_gen) if p_gen_override else bus.p_gen
            total_load += p_load
            if bid != self.slack_bus_id:
                total_gen_nonslack += p_gen
                p_inj[idx] = p_gen - p_load

        # Slack bus balances total generation and load
        slack_idx = bus_idx_map[self.slack_bus_id]
        p_inj[slack_idx] = total_load - total_gen_nonslack

        B_bus, B_branch, A = self.compute_b_bus(active_only=True)
        ref_idx = slack_idx
        non_slack_idx = [i for i in range(n) if i != ref_idx]

        B_reduced = B_bus[np.ix_(non_slack_idx, non_slack_idx)]
        p_inj_reduced_pu = p_inj[non_slack_idx] / self.base_mva

        theta_reduced = np.linalg.solve(B_reduced, p_inj_reduced_pu)

        theta = np.zeros(n, dtype=np.float64)
        for r_i, i in enumerate(non_slack_idx):
            theta[i] = theta_reduced[r_i]

        # Branch power flows: P_ij = (theta_i - theta_j) / x_ij * Base_MVA
        flows = np.zeros(m, dtype=np.float64)
        for lid, br in self.branches.items():
            if br.in_service:
                i = bus_idx_map[br.from_bus]
                j = bus_idx_map[br.to_bus]
                flows[lid] = (theta[i] - theta[j]) / br.x * self.base_mva

        # Restore status
        for lid, status in orig_status.items():
            self.branches[lid].in_service = status

        return flows, theta, True
