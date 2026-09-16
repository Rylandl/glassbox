"""Independent checks of the synthetic qualification experiment's contracts."""

import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    sys.modules.pop("experiment_model_qualification", None)
    import experiment_model_qualification

    return experiment_model_qualification


def test_linear_counterfactual_derivative_and_policy_ambiguity(experiment):
    x0 = np.array([[0.7], [-0.3]])
    u = np.array([[[0.1], [0.2], [-0.1]], [[-0.2], [0.0], [0.4]]])
    plus, minus = u.copy(), u.copy()
    plus[:, 0] += 0.05
    minus[:, 0] -= 0.05
    derivative = (
        experiment.linear_rollout(x0, plus) - experiment.linear_rollout(x0, minus)
    ) / 0.1
    expected = np.broadcast_to(
        (0.2 * 1.08 ** np.arange(3))[None, :, None], derivative.shape
    )
    np.testing.assert_allclose(derivative, expected, rtol=1e-12)
    for b in (0.05, 0.2, 0.5):
        a = 0.98 + 0.5 * b
        np.testing.assert_allclose(a * x0 + b * (-0.5 * x0), 0.98 * x0)


def test_hidden_memory_probe_is_causal_and_indistinguishable(experiment):
    past_x, past_u, future_u, target = experiment.memory_probe()
    np.testing.assert_array_equal(past_x[0, -3:], past_x[1, -3:])
    np.testing.assert_array_equal(past_u[0, -2:], past_u[1, -2:])
    np.testing.assert_array_equal(future_u[0], future_u[1])
    assert past_u[0, 0, 0] == -1 and past_u[1, 0, 0] == 1
    expected = 0.2 * 0.8 ** np.arange(5)
    np.testing.assert_allclose(target[:, :, 0], np.stack((-expected, expected)))
    # Any common prediction has paired squared error >= the squared half-gap.
    candidate = np.array([0.2, -0.1, 0.3, 0.0, -0.2])
    error = np.mean((target[:, :, 0] - candidate) ** 2, axis=0)
    np.testing.assert_allclose(error, expected**2 + candidate**2)


@pytest.mark.parametrize(
    "name", ["identity", "units_offset", "orthogonal", "permuted", "duplicate_first"]
)
def test_encodings_preserve_information_and_common_coordinates(experiment, name):
    transform = experiment.encoding(name)
    states = np.array([[0.2, 0.7], [-1.1, 0.03]])
    np.testing.assert_allclose(
        transform.decode(transform.states(states)), states, atol=1e-12
    )
    np.testing.assert_allclose(
        transform.state_map @ transform.state_inverse, np.eye(2), atol=1e-12
    )
    if name == "orthogonal":
        np.testing.assert_allclose(
            np.linalg.norm(transform.states(states), axis=-1),
            np.linalg.norm(states, axis=-1),
        )
    if name == "duplicate_first":
        np.testing.assert_array_equal(
            transform.states(states)[:, 2:], np.repeat(states[:, :1], 6, axis=1)
        )


def test_recordings_and_windows_keep_future_targets_separate(experiment):
    a = experiment.recordings("encoding", 101)
    b = experiment.recordings("encoding", 101, evaluation=True)
    assert not {s.recording_id for s in a} & {s.recording_id for s in b}
    batch = experiment.windows(b)
    assert batch.past_states.shape == (124, 3, 2)
    np.testing.assert_array_equal(batch.past_states[0], b[0].states[:3])
    np.testing.assert_array_equal(batch.future_inputs[0], b[0].inputs[2:7])
    np.testing.assert_array_equal(batch.future_states[0], b[0].states[3:8])
    # Independent explicit one-step check of the generated nonlinear plant.
    x, u = b[0].states[2], b[0].inputs[2]
    expected = [
        0.9 * x[0] + 0.1 * np.tanh(1.5 * u[0]) + 0.025 * x[1],
        0.86 * x[1] + 0.08 * u[1] + 0.04 * np.tanh(x[0]),
    ]
    np.testing.assert_allclose(batch.future_states[0, 0], expected)
