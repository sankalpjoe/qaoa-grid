"""
Unit tests for GPU Continuous DC-OPF Solver and Bloch Angle Mapping.
"""

import pytest
import numpy as np
import torch

from grid_resilience.grid import GridNetwork
from grid_resilience.gpu_solver import ContinuousDcOpfGpu


def test_gpu_solver_execution():
    net = GridNetwork.create_ieee14(rating_multiplier=1.20)
    solver = ContinuousDcOpfGpu(net, initiating_outages=[6], candidate_lines=[0, 1, 2, 3, 5, 7])
    res = solver.solve(num_epochs=50)

    assert res.solver_time_ms > 0.0
    assert len(res.bloch_angles) == 6

    # Verify bloch angles in valid range [0, pi]
    for lid, theta in res.bloch_angles.items():
        assert 0.0 <= theta <= np.pi
        # With safe-band 0.08, theta cannot be exactly 0 or pi
        assert theta > 0.1
        assert theta < np.pi - 0.1
