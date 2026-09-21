"""
Ising Hamiltonian Formulation and Hardware-Co-Designed PTDF Sparsification.
Converts Transmission Switching Overload Mitigation into an Ising Cost Hamiltonian:
    H_C = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j + offset
Implements Physics-Informed Sensitivity Truncation to ensure vertex degree <= 3,
matching the native connectivity of IBM Eagle (127q) and Heron (133q) heavy-hex architectures.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import networkx as nx

from ..grid.network import GridNetwork


@dataclass
class IsingHamiltonian:
    num_qubits: int
    candidate_lines: List[int]             # Mapping qubit index -> branch_id
    linear_terms: Dict[int, float]          # qubit_i -> h_i
    quadratic_terms: Dict[Tuple[int, int], float]  # (qubit_i, qubit_j) -> J_ij
    offset: float
    interaction_graph: nx.Graph
    max_degree: int
    is_heavy_hex_compatible: bool           # True if max_degree <= 3

    def evaluate_state(self, bitstring: str) -> float:
        """
        Evaluates Ising energy for a given bitstring (e.g. '10110').
        Bit '0' -> Z = +1 (line switched off)
        Bit '1' -> Z = -1 (line closed / in service)
        Note: Qiskit bitstring order is qubit 0 at rightmost or leftmost depending on convention.
        Here we assume bitstring[i] is qubit i.
        """
        z_vals = [1.0 if bit == '0' else -1.0 for bit in bitstring]
        energy = self.offset

        for i, h in self.linear_terms.items():
            energy += h * z_vals[i]

        for (i, j), J in self.quadratic_terms.items():
            energy += J * z_vals[i] * z_vals[j]

        return energy


class IsingModelBuilder:
    """
    Constructs the physical Ising Hamiltonian for Cascading Outage Mitigation.
    """

    def __init__(
        self,
        network: GridNetwork,
        initiating_outages: List[int],
        candidate_lines: Optional[List[int]] = None,
        max_candidates: int = 10
    ):
        self.network = network
        self.initiating_outages = initiating_outages

        # Identify candidate switching lines
        if candidate_lines is None:
            # Select non-bridge lines with significant flow
            g = network.to_networkx()
            bridges = list(nx.bridges(g))
            bridge_lids = set()
            for u, v in bridges:
                bridge_lids.add(g[u][v]['branch_id'])

            # Solve post-contingency flow to find overloaded lines
            status = {lid: (lid not in initiating_outages) for lid in network.branches}
            flows, _, _ = network.solve_dc_power_flow(line_status=status)

            overloaded_lines = [
                lid for lid, br in network.branches.items()
                if lid not in initiating_outages and abs(flows[lid]) > br.rating
            ]
            if not overloaded_lines:
                overloaded_lines = sorted(
                    [l for l in network.branches if l not in initiating_outages],
                    key=lambda l: abs(flows[l]) / network.branches[l].rating,
                    reverse=True
                )[:2]

            scored_lines = []
            for lid in network.branches:
                if lid not in initiating_outages and lid not in bridge_lids:
                    # Test flow shift when opening this line
                    trial_status = dict(status)
                    trial_status[lid] = False
                    trial_flows, _, ok = network.solve_dc_power_flow(line_status=trial_status)
                    if ok:
                        # Relief: sum of flow reduction on overloaded lines
                        relief = sum(abs(flows[ol]) - abs(trial_flows[ol]) for ol in overloaded_lines)
                        max_load = max(abs(trial_flows[l]) / network.branches[l].rating for l in network.branches if trial_status[l])
                        # Score prioritizes valid cures (max_load < 1.0) and high positive relief
                        score = (1000.0 if max_load <= 1.0 else 0.0) + relief * 10.0 - max_load * 5.0
                        scored_lines.append((lid, score))

            scored_lines.sort(key=lambda x: x[1], reverse=True)
            self.candidate_lines = [lid for lid, _ in scored_lines[:max_candidates]]
        else:
            self.candidate_lines = candidate_lines

        self.num_qubits = len(self.candidate_lines)
        self.line_to_qubit = {lid: i for i, lid in enumerate(self.candidate_lines)}

    def compute_lodf_sensitivities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Line Outage Distribution Factors (LODF) or sensitivity of branch flows
        to intentional switching of candidate lines.
        Returns:
            base_flows: (M,) post-contingency line flows
            lodf_matrix: (M, K) sensitivity matrix: delta P_l = LODF_{l, k} * P_k^0
        """
        m = self.network.num_branches
        k_count = self.num_qubits

        # Base post-contingency flows
        status_base = {lid: (lid not in self.initiating_outages) for lid in self.network.branches}
        base_flows, _, _ = self.network.solve_dc_power_flow(line_status=status_base)

        lodf_matrix = np.zeros((m, k_count), dtype=np.float64)

        # For each candidate line, simulate opening it and compute flow difference
        for k_idx, cand_lid in enumerate(self.candidate_lines):
            status_switched = dict(status_base)
            status_switched[cand_lid] = False

            sw_flows, _, ok = self.network.solve_dc_power_flow(line_status=status_switched)
            if ok:
                lodf_matrix[:, k_idx] = sw_flows - base_flows
            else:
                # Disconnection: assign high penalty sensitivity
                lodf_matrix[:, k_idx] = 1000.0

        return base_flows, lodf_matrix

    def build_hamiltonian(
        self,
        penalty_overload: float = 2.0,
        penalty_switch: float = 1.5,
        penalty_island: float = 20.0,
        sparsify_heavy_hex: bool = True,
        max_heavy_hex_degree: int = 3
    ) -> IsingHamiltonian:
        """
        Constructs the Ising Hamiltonian.
        If sparsify_heavy_hex is True, prunes low-magnitude interaction terms
        to enforce max vertex degree <= 3 (compatible with IBM Eagle/Heron lattices).
        """
        base_flows, lodf_matrix = self.compute_lodf_sensitivities()
        k_count = self.num_qubits

        # Find overloaded lines under contingency
        overloaded_lines = []
        for lid, br in self.network.branches.items():
            if lid not in self.initiating_outages and abs(base_flows[lid]) > br.rating:
                overloaded_lines.append(lid)

        # If no line is overloaded under this contingency, take top 2 most stressed
        if not overloaded_lines:
            stress_ranked = sorted(
                [lid for lid in self.network.branches if lid not in self.initiating_outages],
                key=lambda l: abs(base_flows[l]) / self.network.branches[l].rating,
                reverse=True
            )
            overloaded_lines = stress_ranked[:2]

        # Cost in terms of binary variable y_k in {0, 1} where y_k = 1 means switched off, y_k = 0 means kept in service
        # Note: y_k = (1 + Z_k) / 2
        # C(y) = sum_{l in overloads} w_l * (P_l(y) - sign(P_l^0)*Rating_l)^2 + lambda_sw * sum_k y_k
        Q = np.zeros((k_count, k_count), dtype=np.float64)
        c_linear = np.zeros(k_count, dtype=np.float64)
        c_const = 0.0

        for ol_lid in overloaded_lines:
            br = self.network.branches[ol_lid]
            target_sign = np.sign(base_flows[ol_lid]) if abs(base_flows[ol_lid]) > 1e-3 else 1.0
            excess_flow = base_flows[ol_lid] - target_sign * br.rating  # amount exceeding rating
            # P_l(y) - target = excess_flow + sum_k lodf_matrix[ol_lid, k] * y_k
            sens = lodf_matrix[ol_lid, :]

            # Squared error: (excess + sens^T y)^2 = excess^2 + 2*excess*(sens^T y) + y^T (sens sens^T) y
            c_const += penalty_overload * (excess_flow ** 2)
            c_linear += penalty_overload * 2.0 * excess_flow * sens
            Q += penalty_overload * np.outer(sens, sens)

        # Add switching penalty
        c_linear += penalty_switch * np.ones(k_count)

        # Add anti-islanding 2-cut penalties
        g_base = self.network.to_networkx()
        for i in range(k_count):
            for j in range(i + 1, k_count):
                l1 = self.candidate_lines[i]
                l2 = self.candidate_lines[j]
                # Check if cutting both disconnects the grid
                status_test = {lid: (lid not in self.initiating_outages and lid not in (l1, l2)) for lid in self.network.branches}
                _, _, ok = self.network.solve_dc_power_flow(line_status=status_test)
                if not ok:
                    # Penalize simultaneous opening of both lines
                    Q[i, j] += penalty_island
                    Q[j, i] += penalty_island

        # Transform binary QUBO y_i into Ising Z_i:
        # y_i = (1 + Z_i) / 2
        # y_i y_j = (1 + Z_i + Z_j + Z_i Z_j) / 4
        # y_i^2 = y_i = (1 + Z_i) / 2
        h_terms = {}
        J_terms = {}
        offset = c_const

        # Diagonal terms of Q combine with c_linear
        diag_coeffs = np.diag(Q) + c_linear
        for i in range(k_count):
            offset += 0.5 * diag_coeffs[i]
            h_terms[i] = 0.5 * diag_coeffs[i]

        # Off-diagonal terms
        for i in range(k_count):
            for j in range(i + 1, k_count):
                q_ij = Q[i, j] + Q[j, i]
                if abs(q_ij) > 1e-6:
                    # (1 + Z_i + Z_j + Z_i Z_j) / 4 * q_ij
                    offset += 0.25 * q_ij
                    h_terms[i] += 0.25 * q_ij
                    h_terms[j] += 0.25 * q_ij
                    J_terms[(i, j)] = 0.25 * q_ij

        # Build initial interaction graph
        g_interact = nx.Graph()
        for i in range(k_count):
            g_interact.add_node(i)
        for (i, j), J in J_terms.items():
            g_interact.add_edge(i, j, weight=abs(J), J=J)

        # Hardware-Co-Designed Heavy-Hex Sparsification
        if sparsify_heavy_hex:
            # Enforce max degree <= max_heavy_hex_degree (3)
            # Greedy pruning: remove edges with smallest absolute weight |J| until max degree <= 3
            while True:
                degrees = dict(g_interact.degree())
                violating_nodes = [n for n, d in degrees.items() if d > max_heavy_hex_degree]
                if not violating_nodes:
                    break

                # Pick the violating node with highest degree
                target_node = max(violating_nodes, key=lambda n: degrees[n])
                # Find the incident edge with the smallest |J|
                incident_edges = list(g_interact.edges(target_node, data=True))
                incident_edges.sort(key=lambda e: e[2]['weight'])
                # Remove the weakest edge
                edge_to_remove = incident_edges[0]
                g_interact.remove_edge(edge_to_remove[0], edge_to_remove[1])

            # Reconstruct J_terms from pruned interaction graph
            pruned_J_terms = {}
            for u, v, d in g_interact.edges(data=True):
                pair = (min(u, v), max(u, v))
                pruned_J_terms[pair] = d['J']
            J_terms = pruned_J_terms

        max_deg = max([d for _, d in g_interact.degree()]) if g_interact.nodes else 0
        is_hh_compat = bool(max_deg <= max_heavy_hex_degree)

        return IsingHamiltonian(
            num_qubits=k_count,
            candidate_lines=self.candidate_lines,
            linear_terms=h_terms,
            quadratic_terms=J_terms,
            offset=offset,
            interaction_graph=g_interact,
            max_degree=max_deg,
            is_heavy_hex_compatible=is_hh_compat
        )
