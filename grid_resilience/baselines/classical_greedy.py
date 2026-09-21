"""
Classical PTDF-Ranked Greedy Transmission Switching Heuristic.
Serves as the primary classical industry baseline.
Iteratively selects line switching actions that yield the steepest reduction
in peak line overload while verifying grid connectivity.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import numpy as np

from ..grid.network import GridNetwork
from ..grid.cascade_simulator import CascadeSimulator, SwitchingEvaluation


@dataclass
class GreedyResult:
    is_valid: bool
    is_connected: bool
    switched_lines: List[int]
    num_switches: int
    max_loading_pct: float
    thermal_violation_mw: float
    total_cost: float
    computation_time_ms: float
    step_history: List[Dict]


class ClassicalPtdfGreedySolver:
    """
    Classical greedy heuristic prioritizing transmission switching based on
    PTDF/LODF sensitivity to relieve line overloads without bus islanding.
    """

    def __init__(
        self,
        network: GridNetwork,
        initiating_outages: List[int],
        candidate_lines: Optional[List[int]] = None,
        max_switching_budget: int = 3
    ):
        self.network = network
        self.initiating_outages = initiating_outages
        self.max_budget = max_switching_budget
        self.simulator = CascadeSimulator(network)

        if candidate_lines is None:
            self.candidate_lines = [lid for lid in network.branches if lid not in initiating_outages]
        else:
            self.candidate_lines = [lid for lid in candidate_lines if lid not in initiating_outages]

    def solve(self) -> GreedyResult:
        """
        Executes the greedy transmission switching search.
        """
        start_time = time.perf_counter()

        current_switches = []
        step_history = []

        # Evaluate initial state
        initial_eval = self.simulator.evaluate_switching_action(self.initiating_outages, {})
        step_history.append({
            "step": 0,
            "switches": [],
            "max_loading_pct": initial_eval.max_loading_pct,
            "overloads": list(initial_eval.overloaded_lines),
            "is_valid": initial_eval.is_valid
        })

        if initial_eval.is_valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GreedyResult(
                is_valid=True,
                is_connected=True,
                switched_lines=[],
                num_switches=0,
                max_loading_pct=initial_eval.max_loading_pct,
                thermal_violation_mw=0.0,
                total_cost=0.0,
                computation_time_ms=elapsed_ms,
                step_history=step_history
            )

        best_eval = initial_eval

        for step in range(1, self.max_budget + 1):
            best_cand = None
            best_cand_eval = None

            # Test each unswitched candidate
            for cand_lid in self.candidate_lines:
                if cand_lid in current_switches:
                    continue

                trial_switches = {lid: 0 for lid in current_switches + [cand_lid]}
                ev = self.simulator.evaluate_switching_action(self.initiating_outages, trial_switches)

                # Must preserve connectivity
                if not ev.is_connected:
                    continue

                # We look for reduction in thermal violation or max loading
                if best_cand_eval is None or ev.total_cost < best_cand_eval.total_cost:
                    best_cand = cand_lid
                    best_cand_eval = ev

            if best_cand is None or best_cand_eval.total_cost >= best_eval.total_cost:
                # No further improvement found
                break

            current_switches.append(best_cand)
            best_eval = best_cand_eval

            step_history.append({
                "step": step,
                "switches": list(current_switches),
                "max_loading_pct": best_eval.max_loading_pct,
                "overloads": list(best_eval.overloaded_lines),
                "is_valid": best_eval.is_valid
            })

            if best_eval.is_valid:
                # Valid solution found, arrest search
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return GreedyResult(
            is_valid=best_eval.is_valid,
            is_connected=best_eval.is_connected,
            switched_lines=current_switches,
            num_switches=len(current_switches),
            max_loading_pct=best_eval.max_loading_pct,
            thermal_violation_mw=best_eval.thermal_violation_mw,
            total_cost=best_eval.total_cost,
            computation_time_ms=elapsed_ms,
            step_history=step_history
        )
