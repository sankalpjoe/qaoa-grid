"""
Cascading Outage Simulator and Transmission Switching Evaluation.
Simulates contingency initiation, overload propagation, cascading failure loops,
and evaluates corrective transmission switching candidates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import networkx as nx
from .network import GridNetwork


@dataclass
class OutageState:
    step: int
    tripped_lines: List[int]
    overloaded_lines: List[int]
    max_loading_pct: float
    is_connected: bool
    blackout: bool = False


@dataclass
class SwitchingEvaluation:
    is_valid: bool                      # Connected and zero overloads
    is_connected: bool                  # Graph remains intact
    max_loading_pct: float              # Max line loading % (e.g. 95% = 0.95)
    overloaded_lines: List[int]         # Lines exceeding 100% capacity
    num_switches: int                   # Number of lines intentionally switched out
    switched_out_lines: List[int]       # IDs of switched lines
    thermal_violation_mw: float         # Sum of MW exceeding thermal limits
    total_cost: float                   # Objective value for optimization


class CascadeSimulator:
    """
    Simulates cascading failure dynamics following an initial N-1 / N-k contingency
    and assesses effectiveness of transmission switching actions.
    """

    def __init__(self, network: GridNetwork, overload_trip_threshold: float = 1.0):
        self.network = network
        self.overload_trip_threshold = overload_trip_threshold  # 1.0 = 100% rating

    def simulate_unmitigated_cascade(
        self,
        initiating_outages: List[int],
        max_steps: int = 10
    ) -> List[OutageState]:
        """
        Simulates an unmitigated cascading blackout starting from initiating outages.
        Overloaded lines trip sequentially in each step.
        """
        history = []
        current_tripped = set(initiating_outages)

        for step in range(max_steps):
            # Apply tripped lines
            status = {lid: (lid not in current_tripped) for lid in self.network.branches}
            flows, angles, connected = self.network.solve_dc_power_flow(line_status=status)

            if not connected:
                history.append(OutageState(
                    step=step,
                    tripped_lines=sorted(list(current_tripped)),
                    overloaded_lines=[],
                    max_loading_pct=999.0,
                    is_connected=False,
                    blackout=True
                ))
                break

            # Check for line overloads
            overloads = []
            max_load_pct = 0.0
            for lid, br in self.network.branches.items():
                if br.in_service and lid not in current_tripped:
                    flow_abs = abs(flows[lid])
                    pct = flow_abs / br.rating
                    if pct > max_load_pct:
                        max_load_pct = pct
                    if pct > self.overload_trip_threshold:
                        overloads.append(lid)

            state = OutageState(
                step=step,
                tripped_lines=sorted(list(current_tripped)),
                overloaded_lines=overloads,
                max_loading_pct=max_load_pct,
                is_connected=True,
                blackout=(len(overloads) == 0 and len(current_tripped) > len(initiating_outages))
            )
            history.append(state)

            if not overloads:
                # Cascade arrested
                break

            # Overloaded lines trip in next step
            new_trips = set(overloads) - current_tripped
            if not new_trips:
                break
            current_tripped.update(new_trips)

        return history

    def evaluate_switching_action(
        self,
        initiating_outages: List[int],
        switching_decisions: Dict[int, int],
        penalty_violation: float = 500.0,
        penalty_switch: float = 10.0,
        penalty_disconnected: float = 10000.0
    ) -> SwitchingEvaluation:
        """
        Evaluates a candidate transmission switching configuration.
        switching_decisions: mapping branch_id -> 1 (remain closed) or 0 (intentionally open/switch).
        """
        switched_out = [lid for lid, z in switching_decisions.items() if z == 0 and lid not in initiating_outages]
        num_switches = len(switched_out)

        # Build full status dictionary
        status = {}
        for lid in self.network.branches:
            if lid in initiating_outages:
                status[lid] = False
            elif lid in switching_decisions:
                status[lid] = bool(switching_decisions[lid] == 1)
            else:
                status[lid] = True

        flows, angles, connected = self.network.solve_dc_power_flow(line_status=status)

        if not connected:
            return SwitchingEvaluation(
                is_valid=False,
                is_connected=False,
                max_loading_pct=999.0,
                overloaded_lines=[],
                num_switches=num_switches,
                switched_out_lines=switched_out,
                thermal_violation_mw=1000.0,
                total_cost=penalty_disconnected + num_switches * penalty_switch
            )

        overloaded = []
        max_load = 0.0
        thermal_violation_mw = 0.0

        for lid, br in self.network.branches.items():
            if status[lid]:
                flow_abs = abs(flows[lid])
                pct = flow_abs / br.rating
                if pct > max_load:
                    max_load = pct
                if flow_abs > br.rating:
                    overloaded.append(lid)
                    thermal_violation_mw += (flow_abs - br.rating)

        is_valid = (len(overloaded) == 0) and connected
        total_cost = (
            penalty_violation * (thermal_violation_mw ** 2)
            + penalty_switch * num_switches
        )

        return SwitchingEvaluation(
            is_valid=is_valid,
            is_connected=connected,
            max_loading_pct=max_load,
            overloaded_lines=overloaded,
            num_switches=num_switches,
            switched_out_lines=switched_out,
            thermal_violation_mw=thermal_violation_mw,
            total_cost=total_cost
        )
