"""
Unit tests for Grid Physics, DC Power Flow, and Cascade Simulation.
"""

import pytest
import numpy as np
import networkx as nx

from grid_resilience.grid import GridNetwork, CascadeSimulator


def test_ieee14_topology():
    net = GridNetwork.create_ieee14()
    assert net.num_buses == 14
    assert net.num_branches == 20
    assert net.is_connected(active_only=True)


def test_b_matrix_symmetry():
    net = GridNetwork.create_ieee14()
    B_bus, B_branch, A = net.compute_b_bus()
    assert B_bus.shape == (14, 14)
    # B_bus must be symmetric
    assert np.allclose(B_bus, B_bus.T, atol=1e-8)


def test_dc_power_flow_balance():
    net = GridNetwork.create_ieee14()
    flows, angles, ok = net.solve_dc_power_flow()
    assert ok is True
    assert len(flows) == 20
    assert len(angles) == 14
    # Slack bus angle must be 0
    assert abs(angles[net.slack_bus_id - 1]) < 1e-10
    # Line flows must be non-trivial
    assert np.max(np.abs(flows)) > 50.0


def test_ptdf_calculation():
    net = GridNetwork.create_ieee14()
    ptdf = net.compute_ptdf()
    assert ptdf.shape == (20, 14)


def test_cascade_simulator_outage():
    net = GridNetwork.create_ieee14(rating_multiplier=1.0)
    sim = CascadeSimulator(net)
    history = sim.simulate_unmitigated_cascade(initiating_outages=[0])
    assert len(history) >= 2
    # Tripping line 0 in standard unscaled network causes overloads
    assert len(history[0].overloaded_lines) > 0
