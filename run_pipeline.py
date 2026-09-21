"""
Master Execution Pipeline for IEEE Smart Grid Resilience Study.
Executes end-to-end benchmark:
RTX 5070 GPU Continuous OPF -> HG-WS-QAOA & Standard QAOA -> Heavy-Hex Transpilation -> Aer Noisy Simulation -> Evaluation & Plots.
"""

import argparse
import sys
import os
import json

from grid_resilience.grid import GridNetwork, CascadeSimulator
from grid_resilience.benchmarks import BenchmarkEvaluator, PublicationPlotter


def main():
    parser = argparse.ArgumentParser(description="Run Graph-Constrained WS-QAOA Pipeline")
    parser.add_argument("--network", type=str, default="ieee14", choices=["ieee14", "ieee30"], help="Grid network model")
    parser.add_argument("--outage", type=int, default=6, help="Branch ID to trip as initial contingency")
    parser.add_argument("--candidates", type=int, default=8, help="Number of candidate switching lines (qubits)")
    parser.add_argument("--shots", type=int, default=4096, help="Measurement shots per quantum execution")
    parser.add_argument("--plots", action="store_true", default=True, help="Generate publication-grade figures")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Output directory for reports and figures")

    args = parser.parse_args()

    print("================================================================================")
    print(" HARDWARE-ALIGNED GRAPH-CONSTRAINED WARM-STARTED QAOA (HG-WS-QAOA)")
    print(" Cascading Outage Mitigation in Smart Transmission Grids")
    print(" Hardware Split: NVIDIA RTX 5070 (DC-OPF) + IBM Heavy-Hex (Qiskit QAOA)")
    print("================================================================================\n")

    # 1. Load Network
    if args.network == "ieee14":
        net = GridNetwork.create_ieee14(rating_multiplier=1.20)
    else:
        net = GridNetwork.create_ieee30(rating_multiplier=1.20)

    print(f"Network: {net.name} ({net.num_buses} Buses, {net.num_branches} Branches)")

    # 2. Run Evaluator
    evaluator = BenchmarkEvaluator(
        network=net,
        initiating_outages=[args.outage],
        max_candidates=args.candidates,
        shots=args.shots,
        seed=42
    )

    report = evaluator.run_full_benchmark()

    # 3. Print Structured Results Table
    print("\n" + "=" * 95)
    print(f"{'Algorithm / Method':<24} | {'Noise':<8} | {'P(valid)':<10} | {'Exp. Cost':<11} | {'AR':<6} | {'2Q Gates':<8} | {'Depth':<6}")
    print("=" * 95)

    # Random baseline
    print(f"{'Uniform Random':<24} | {'Ideal':<8} | {report.random_sampling_p_valid*100.0:>8.2f}% | {'N/A':<11} | {'N/A':<6} | {'0':<8} | {'0':<6}")

    # Greedy baseline
    greedy_status = "100.00%" if report.greedy_result.is_valid else "0.00%"
    print(f"{'Classical Greedy (PTDF)':<24} | {'Classical':<8} | {greedy_status:>10} | {report.greedy_result.total_cost:>11.2f} | {'1.00':<6} | {'N/A':<8} | {'N/A':<6}")

    # Quantum evaluations
    for ev in report.evaluations:
        noise_str = "Ideal" if ev.noise_level == 0.0 else "IBM-Hex"
        print(f"{ev.method_name:<24} | {noise_str:<8} | {ev.p_valid*100.0:>8.2f}% | {ev.expected_cost:>11.2f} | {ev.approximation_ratio:>6.3f} | {ev.transpiled_2q_gates:<8} | {ev.transpiled_depth:<6}")

    print("=" * 95)

    # 4. Generate Figures
    if args.plots:
        plotter = PublicationPlotter(output_dir=args.reports_dir)
        f1 = plotter.plot_sampling_validity(report)
        f2 = plotter.plot_noise_degradation(report)
        f3 = plotter.plot_circuit_depth_comparison(report)
        print(f"\nGenerated Publication Figures in '{args.reports_dir}/':")
        print(f"  - {f1}")
        print(f"  - {f2}")
        print(f"  - {f3}")

    # 5. Save JSON summary
    summary_path = os.path.join(args.reports_dir, "benchmark_summary.json")
    summary_data = {
        "network": report.network_name,
        "num_qubits": report.num_qubits,
        "candidate_lines": report.candidate_lines,
        "initiating_outages": report.initiating_outages,
        "ground_truth_min_cost": report.ground_truth_min_cost,
        "ground_truth_valid_count": len(report.ground_truth_valid_states),
        "random_p_valid": report.random_sampling_p_valid,
        "gpu_solver_time_ms": report.gpu_solver_result.solver_time_ms,
        "gpu_device": report.gpu_solver_result.device_used,
        "results": [
            {
                "method": e.method_name,
                "noise": e.noise_level,
                "p_valid": e.p_valid,
                "expected_cost": e.expected_cost,
                "min_cost": e.min_cost_sampled,
                "ar": e.approximation_ratio,
                "depth": e.transpiled_depth,
                "two_qubit_gates": e.transpiled_2q_gates,
                "swap_estimate": e.swap_count_estimate,
                "time_ms": e.execution_time_ms
            }
            for e in report.evaluations
        ]
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved JSON benchmark summary to: {summary_path}")


if __name__ == "__main__":
    main()
