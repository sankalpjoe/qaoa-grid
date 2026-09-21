"""
Comprehensive Benchmarking and Evaluation Engine.
Implements rigorous comparative analysis across:
1. Hardware-Aligned Graph-Constrained Warm-Started QAOA (HG-WS-QAOA, p=1, 2)
2. Standard Un-Warm-Started QAOA (p=1, 2)
3. Classical PTDF Greedy Heuristic
4. Exact Ground-Truth Enumeration
Evaluates Deliverable Metrics:
- Zero-Violation Valid Sampling Probability P(valid)
- Approximation Ratio (AR)
- Physical Noise Degradation Profile (Ideal vs. Calibrated IBM Heavy-Hex Noise)
- Circuit Transpilation Footprint (Depth, SWAPs, 2Q Gates).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from ..grid.network import GridNetwork
from ..grid.cascade_simulator import CascadeSimulator, SwitchingEvaluation
from ..gpu_solver.dc_opf_gpu import ContinuousDcOpfGpu, GpuRelaxationResult
from ..hamiltonian.ising_formulation import IsingModelBuilder, IsingHamiltonian
from ..quantum.warm_start_qaoa import WarmStartQaoaCircuit
from ..quantum.standard_qaoa import StandardQaoaCircuit
from ..quantum.heavy_hex_mapping import HeavyHexMapper, TranspilationResult
from ..quantum.noise_models import HardwareNoiseFactory
from ..baselines.classical_greedy import ClassicalPtdfGreedySolver, GreedyResult


@dataclass
class MethodEvaluationResult:
    method_name: str
    is_quantum: bool
    depth_p: Optional[int]
    noise_level: float                 # 0.0 = ideal, 1.0 = baseline hardware noise
    p_valid: float                     # Fraction of valid zero-violation samples [0, 1]
    expected_cost: float
    min_cost_sampled: float
    approximation_ratio: float
    transpiled_2q_gates: int
    transpiled_depth: int
    swap_count_estimate: int
    execution_time_ms: float
    sample_distribution: Dict[str, int]
    valid_bitstrings_found: List[str]


@dataclass
class ComprehensiveBenchmarkReport:
    network_name: str
    initiating_outages: List[int]
    candidate_lines: List[int]
    num_qubits: int
    ground_truth_min_cost: float
    ground_truth_valid_states: List[str]
    random_sampling_p_valid: float
    gpu_solver_result: GpuRelaxationResult
    greedy_result: GreedyResult
    evaluations: List[MethodEvaluationResult]
    noise_sweep_data: Dict[str, List[Tuple[float, float]]]  # method -> list of (noise_scale, p_valid)


class BenchmarkEvaluator:
    """
    Orchestrates the end-to-end benchmark comparison.
    """

    def __init__(
        self,
        network: GridNetwork,
        initiating_outages: List[int],
        max_candidates: int = 8,
        shots: int = 8192,
        seed: int = 42
    ):
        self.network = network
        self.initiating_outages = initiating_outages
        self.max_candidates = max_candidates
        self.shots = shots
        self.seed = seed
        self.simulator = CascadeSimulator(network)

        # Build Ising model
        self.model_builder = IsingModelBuilder(
            network=network,
            initiating_outages=initiating_outages,
            max_candidates=max_candidates
        )
        self.candidate_lines = self.model_builder.candidate_lines
        self.num_qubits = len(self.candidate_lines)

        # Build sparse (heavy-hex aligned) and dense Hamiltonians
        self.h_sparse = self.model_builder.build_hamiltonian(sparsify_heavy_hex=True, max_heavy_hex_degree=3)
        self.h_dense = self.model_builder.build_hamiltonian(sparsify_heavy_hex=False)

        # Heavy-Hex mapper
        self.mapper = HeavyHexMapper(architecture="ibm_heron", lattice_distance=3)

    def compute_ground_truth(self) -> Tuple[float, List[str], Dict[str, float]]:
        """
        Enumerates all 2^K configurations to find exact minimum cost and all valid states.
        """
        all_costs = {}
        valid_states = []
        min_cost = float("inf")

        for idx in range(2 ** self.num_qubits):
            # Bitstring format: '0' means line opened, '1' means line kept closed
            raw_bits = format(idx, f"0{self.num_qubits}b")
            # Mapping bitstring to line decisions: qubit i corresponds to candidate_lines[i]
            decisions = {}
            for i, bit in enumerate(raw_bits):
                lid = self.candidate_lines[i]
                decisions[lid] = int(bit)

            ev = self.simulator.evaluate_switching_action(self.initiating_outages, decisions)
            all_costs[raw_bits] = ev.total_cost

            if ev.is_valid:
                valid_states.append(raw_bits)

            if ev.total_cost < min_cost:
                min_cost = ev.total_cost

        return min_cost, valid_states, all_costs

    def evaluate_circuit_samples(
        self,
        counts: Dict[str, int],
        all_costs: Dict[str, float],
        valid_states: List[str],
        ground_truth_min_cost: float
    ) -> Tuple[float, float, float, float, List[str]]:
        """
        Computes deliverable metrics from measurement counts.
        """
        total_shots = sum(counts.values())
        valid_shots = 0
        total_weighted_cost = 0.0
        min_cost_sampled = float("inf")
        valid_found = set()

        valid_set = set(valid_states)

        for bitstring, count in counts.items():
            # Qiskit measurement returns little-endian (qubit 0 at the end).
            # Reverse so that clean_bits[i] corresponds to qubit i (candidate_lines[i])
            clean_bits = bitstring.replace(" ", "")[::-1]
            cost = all_costs.get(clean_bits, 10000.0)

            total_weighted_cost += cost * count
            if cost < min_cost_sampled:
                min_cost_sampled = cost

            if clean_bits in valid_set:
                valid_shots += count
                valid_found.add(clean_bits)

        p_valid = valid_shots / total_shots if total_shots > 0 else 0.0
        expected_cost = total_weighted_cost / total_shots if total_shots > 0 else 10000.0

        # Approximation ratio on valid states: GT_min / E[Cost | valid]
        if valid_shots > 0:
            valid_cost_sum = sum(all_costs.get(b.replace(" ", "")[::-1], 10000.0) * c for b, c in counts.items() if b.replace(" ", "")[::-1] in valid_set)
            avg_valid_cost = valid_cost_sum / valid_shots
            ar_valid = min(1.0, ground_truth_min_cost / avg_valid_cost) if avg_valid_cost > 0 else 1.0
        else:
            ar_valid = 0.0

        approx_ratio = ar_valid

        return p_valid, expected_cost, min_cost_sampled, approx_ratio, sorted(list(valid_found))

    def run_full_benchmark(
        self,
        noise_scales: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0]
    ) -> ComprehensiveBenchmarkReport:
        """
        Executes the end-to-end benchmark comparison.
        """
        print(f"=== Starting Comprehensive Benchmark on {self.network.name} ===")
        print(f"Initiating Outages: {self.initiating_outages}")
        print(f"Candidate Lines ({self.num_qubits} qubits): {self.candidate_lines}")

        # 1. Ground Truth
        t0 = time.perf_counter()
        gt_min_cost, gt_valid_states, all_costs = self.compute_ground_truth()
        gt_time_ms = (time.perf_counter() - t0) * 1000.0
        p_random = len(gt_valid_states) / (2 ** self.num_qubits)
        print(f"Ground Truth: Min Cost = {gt_min_cost:.2f}, Valid States = {len(gt_valid_states)} / {2**self.num_qubits} ({p_random*100:.2f}%)")

        # 2. Classical Greedy
        greedy_solver = ClassicalPtdfGreedySolver(
            network=self.network,
            initiating_outages=self.initiating_outages,
            candidate_lines=self.candidate_lines
        )
        greedy_res = greedy_solver.solve()
        print(f"Classical Greedy: Valid = {greedy_res.is_valid}, Cost = {greedy_res.total_cost:.2f}, Switched = {greedy_res.switched_lines}, Time = {greedy_res.computation_time_ms:.2f} ms")

        # 3. Classical GPU Solver (RTX 5070)
        gpu_solver = ContinuousDcOpfGpu(
            network=self.network,
            initiating_outages=self.initiating_outages,
            candidate_lines=self.candidate_lines,
            epsilon_safe_band=0.08
        )
        gpu_res = gpu_solver.solve(num_epochs=200)
        print(f"RTX 5070 GPU Continuous Relaxation: Device = {gpu_res.device_used}, Time = {gpu_res.solver_time_ms:.2f} ms")

        evaluations: List[MethodEvaluationResult] = []

        # Optimal angles heuristic initialization for QAOA parameters
        gamma_opt = [0.35, 0.25]
        beta_opt = [0.45, 0.35]

        # 4. Transpilation analysis: Dense vs Hardware-Aligned Sparse
        ws_qaoa_p1 = WarmStartQaoaCircuit(self.h_sparse, gpu_res.bloch_angles, depth_p=1)
        qc_sparse_p1, _, _ = ws_qaoa_p1.build_circuit(gamma_values=[gamma_opt[0]], beta_values=[beta_opt[0]])

        ws_qaoa_dense_p1 = WarmStartQaoaCircuit(self.h_dense, gpu_res.bloch_angles, depth_p=1)
        qc_dense_p1, _, _ = ws_qaoa_dense_p1.build_circuit(gamma_values=[gamma_opt[0]], beta_values=[beta_opt[0]])

        t_res_sparse = self.mapper.transpile_circuit(qc_sparse_p1)
        t_res_dense = self.mapper.transpile_circuit(qc_dense_p1)

        print(f"Heavy-Hex Transpilation: Dense 2Q Gates = {t_res_dense.two_qubit_gate_count}, Sparse 2Q Gates = {t_res_sparse.two_qubit_gate_count} (Reduction: {(1 - t_res_sparse.two_qubit_gate_count/max(1, t_res_dense.two_qubit_gate_count))*100:.1f}%)")
        print(f"Heavy-Hex Depth: Dense Depth = {t_res_dense.transpiled_depth}, Sparse Depth = {t_res_sparse.transpiled_depth} (Reduction: {(1 - t_res_sparse.transpiled_depth/max(1, t_res_dense.transpiled_depth))*100:.1f}%)")

        noise_sweep_data: Dict[str, List[Tuple[float, float]]] = {
            "HG-WS-QAOA (p=1)": [],
            "HG-WS-QAOA (p=2)": [],
            "Standard QAOA (p=1)": [],
            "Standard QAOA (p=2)": []
        }

        # 5. Evaluate methods across noise scales
        for noise_scale in noise_scales:
            noise_model = HardwareNoiseFactory.create_ibm_heavy_hex_noise(noise_scale=noise_scale)
            aer_sim = AerSimulator(noise_model=noise_model if noise_scale > 0 else None)

            # Test configurations:
            configs = [
                ("HG-WS-QAOA (p=1)", True, 1, self.h_sparse, True),
                ("HG-WS-QAOA (p=2)", True, 2, self.h_sparse, True),
                ("Standard QAOA (p=1)", False, 1, self.h_sparse, False),
                ("Standard QAOA (p=2)", False, 2, self.h_sparse, False),
            ]

            for name, is_ws, p, h_model, use_ws in configs:
                t_exec_start = time.perf_counter()

                if use_ws:
                    circuit_gen = WarmStartQaoaCircuit(h_model, gpu_res.bloch_angles, depth_p=p)
                    qc, _, _ = circuit_gen.build_circuit(gamma_values=gamma_opt[:p], beta_values=beta_opt[:p])
                else:
                    circuit_gen = StandardQaoaCircuit(h_model, depth_p=p)
                    qc, _, _ = circuit_gen.build_circuit(gamma_values=gamma_opt[:p], beta_values=beta_opt[:p])

                # Transpile for execution
                t_res = self.mapper.transpile_circuit(qc)
                transpiled_qc = t_res.circuit

                # Run simulation
                result = aer_sim.run(transpiled_qc, shots=self.shots, seed_simulator=self.seed).result()
                counts = result.get_counts()
                exec_time = (time.perf_counter() - t_exec_start) * 1000.0

                p_val, exp_cost, min_cost, ar, val_fnd = self.evaluate_circuit_samples(
                    counts=counts,
                    all_costs=all_costs,
                    valid_states=gt_valid_states,
                    ground_truth_min_cost=gt_min_cost
                )

                noise_sweep_data[name].append((noise_scale, p_val))

                # Record full result for ideal (0.0) and standard hardware noise (1.0)
                if noise_scale in (0.0, 1.0):
                    evaluations.append(MethodEvaluationResult(
                        method_name=name,
                        is_quantum=True,
                        depth_p=p,
                        noise_level=noise_scale,
                        p_valid=p_val,
                        expected_cost=exp_cost,
                        min_cost_sampled=min_cost,
                        approximation_ratio=ar,
                        transpiled_2q_gates=t_res.two_qubit_gate_count,
                        transpiled_depth=t_res.transpiled_depth,
                        swap_count_estimate=t_res.swap_count_estimate,
                        execution_time_ms=exec_time,
                        sample_distribution=counts,
                        valid_bitstrings_found=val_fnd
                    ))

        return ComprehensiveBenchmarkReport(
            network_name=self.network.name,
            initiating_outages=self.initiating_outages,
            candidate_lines=self.candidate_lines,
            num_qubits=self.num_qubits,
            ground_truth_min_cost=gt_min_cost,
            ground_truth_valid_states=gt_valid_states,
            random_sampling_p_valid=p_random,
            gpu_solver_result=gpu_res,
            greedy_result=greedy_res,
            evaluations=evaluations,
            noise_sweep_data=noise_sweep_data
        )
