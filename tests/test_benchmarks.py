"""
Unit tests for Benchmarking Evaluator and Metrics Engine.
"""

import pytest
from grid_resilience.grid import GridNetwork
from grid_resilience.benchmarks import BenchmarkEvaluator


def test_benchmark_evaluator_fast_run():
    net = GridNetwork.create_ieee14(rating_multiplier=1.20)
    evaluator = BenchmarkEvaluator(
        network=net,
        initiating_outages=[6],
        max_candidates=5,
        shots=256,
        seed=42
    )

    report = evaluator.run_full_benchmark(noise_scales=[0.0, 1.0])

    assert report.num_qubits == 5
    assert len(report.evaluations) > 0
    assert 0.0 <= report.random_sampling_p_valid <= 1.0

    for ev in report.evaluations:
        assert 0.0 <= ev.p_valid <= 1.0
        assert ev.approximation_ratio > 0.0
        assert ev.transpiled_depth > 0
