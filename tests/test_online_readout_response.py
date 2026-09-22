"""The saved counterfactual probe uses the same command perturbation as the plant."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_online_readout import response_jacobian, response_scores


def test_command_response_handles_interior_and_clipped_endpoints():
    matrix = np.arange(12, dtype=float).reshape(3, 4) / 10

    class Session:
        def predict(self, past, inputs, commands):
            state = np.zeros((1, 15))
            state[0, 3:6] = matrix @ commands[0]
            return state

    for command in (np.array([0.5, 0.2, 0.8, 0.5]), np.array([0.0, 1.0, 0.0, 1.0])):
        measured = response_jacobian(Session(), None, None, command, 0.01)
        np.testing.assert_allclose(measured, matrix, atol=1e-12)
        assert response_scores(matrix[None], measured[None])["mean_relative_error"] < 1e-12
