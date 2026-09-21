"""
Unit tests for Quantum Circuits, Heavy-Hex Transpilation, and Noise Factory.
"""

import pytest
import numpy as np

from grid_resilience.grid import GridNetwork
from grid_resilience.hamiltonian import IsingModelBuilder
from grid_resilience.quantum import (
    WarmStartQaoaCircuit,
    StandardQaoaCircuit,
    HeavyHexMapper,
    HardwareNoiseFactory
)


def test_quantum_circuit_generation():
    net = GridNetwork.create_ieee14(rating_multiplier=1.20)
    builder = IsingModelBuilder(net, initiating_outages=[6], max_candidates=6)
    h_model = builder.build_hamiltonian(sparsify_heavy_hex=True, max_heavy_hex_degree=3)

    thetas = {lid: np.pi * 0.7 for lid in h_model.candidate_lines}

    # WS-QAOA circuit
    ws_circuit = WarmStartQaoaCircuit(h_model, thetas, depth_p=1)
    qc_ws, _, _ = ws_circuit.build_circuit(gamma_values=[0.3], beta_values=[0.4])
    assert qc_ws.num_qubits == 6

    # Standard QAOA circuit
    std_circuit = StandardQaoaCircuit(h_model, depth_p=1)
    qc_std, _, _ = std_circuit.build_circuit(gamma_values=[0.3], beta_values=[0.4])
    assert qc_std.num_qubits == 6


def test_heavy_hex_transpilation():
    net = GridNetwork.create_ieee14(rating_multiplier=1.20)
    builder = IsingModelBuilder(net, initiating_outages=[6], max_candidates=6)
    h_model = builder.build_hamiltonian(sparsify_heavy_hex=True, max_heavy_hex_degree=3)
    thetas = {lid: np.pi * 0.7 for lid in h_model.candidate_lines}

    ws_circuit = WarmStartQaoaCircuit(h_model, thetas, depth_p=1)
    qc_ws, _, _ = ws_circuit.build_circuit(gamma_values=[0.3], beta_values=[0.4])

    mapper = HeavyHexMapper(architecture="ibm_heron", lattice_distance=3)
    t_res = mapper.transpile_circuit(qc_ws)

    assert t_res.transpiled_depth > 0
    assert t_res.two_qubit_gate_count >= 0
    # Must use basis gates including CZ or ECR
    assert "cz" in t_res.circuit.count_ops() or "ecr" in t_res.circuit.count_ops() or t_res.two_qubit_gate_count == 0


def test_noise_model_generation():
    nm = HardwareNoiseFactory.create_ibm_heavy_hex_noise(noise_scale=1.0)
    assert len(nm.noise_instructions) > 0
