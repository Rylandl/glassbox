"""Independent small algebra checks for the frozen, no-fit support diagnostic."""

import numpy as np
import pytest
from diagnose_online_support import (
    analyze_design,
    original_polynomial,
    physical_coefficients,
    physical_design,
    support_rows,
    svd_design,
)


def fixture():
    rng = np.random.default_rng(932)
    current, delay, memory = 15, 2, 4
    sampled = current * (delay + 1) + memory
    params = dict(
        linear=rng.normal(0, 0.2, (sampled, 6)),
        quadratic=rng.normal(0, 0.01, (current * (current + 1) // 2, 6)),
        bias=rng.normal(size=6),
        memory=np.zeros((sampled, memory)),
    )
    norms = dict(
        body_scale=np.array([2, 0.3, 0.8, 0.1, 0.2, 0.5, 0.7, 0.005, 0.0012]),
        body_mean=np.array([18, 0, 0, 0, 0, 0, -0.15, 0.001, -0.98]),
        input_mean=np.array([0.45, 0.01, -0.1]),
        input_scale=np.array([0.5, 0.02, 5]),
        feature_scale=rng.uniform(0.5, 2, sampled),
        quadratic_scale=rng.uniform(0.5, 2, current * (current + 1) // 2),
        output_scale=rng.uniform(0.5, 2, 6),
    )
    return rng, params, norms, current, delay, memory


def test_physical_transform_preserves_nonzero_centered_quadratic_head():
    rng, params, norms, current, delay, memory = fixture()
    features = rng.normal(size=(11, current))
    history = rng.normal(size=(11, delay, current))
    hidden = np.tanh(rng.normal(size=(11, memory)))
    affine, quadratic = physical_design(features, history, hidden, norms)
    beta_a, beta_q = physical_coefficients(params, norms)
    # Literal independent original feature construction, with nonzero gravity shifts.
    old_sampled = np.c_[features, (history - features[:, None]).reshape(11, -1), hidden]
    old_quadratic = np.array(
        [
            [row[i] * row[k] for i in range(current) for k in range(i, current)]
            for row in features
        ]
    )
    expected = (
        old_sampled / norms["feature_scale"] @ params["linear"]
        + old_quadratic / norms["quadratic_scale"] @ params["quadratic"]
        + params["bias"]
    ) * norms["output_scale"]
    np.testing.assert_allclose(
        affine @ beta_a + quadratic @ beta_q, expected, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        original_polynomial(params, norms, features, history, hidden),
        expected,
        rtol=1e-14,
        atol=1e-14,
    )
    # Gravity has its physical unit scale; motion remains the supported coordinate.
    np.testing.assert_allclose(
        affine[:, 7:10],
        features[:, 6:9] * norms["body_scale"][6:9] + norms["body_mean"][6:9],
    )
    np.testing.assert_allclose(
        affine[:, 1:7], features[:, :6] * norms["body_scale"][:6]
    )


def test_full_row_rank_affine_design_leaves_exactly_no_independent_quadratic_rows():
    affine = np.array([[1, 0, 0], [1, 1, 0]], float)
    quadratic = np.array([[0, 0], [1, 1]], float)
    query_a = np.array([[1, 0, 1]], float)
    query_q = np.array([[3, -2]], float)
    beta_a = np.zeros((3, 6))
    beta_a[2] = np.arange(1, 7)
    beta_q = np.ones((2, 6))
    summary, saved = analyze_design(
        affine, quadratic, query_a, query_q, np.array([0.25, 0.75]), beta_a, beta_q
    )
    conditional = summary["conditional_quadratic"]
    assert summary["affine"]["rank"] == 2
    assert conditional["independent_rows"] == conditional["rank"] == 0
    assert saved["conditional_design"].shape == (0, 2)
    assert conditional["query_support"][0][
        "right_null_energy_fraction"
    ] == pytest.approx(1)
    np.testing.assert_allclose(
        saved["conditional_acceleration"],
        saved["conditional_unsupported_acceleration"],
        atol=1e-13,
    )
    np.testing.assert_allclose(
        saved["retained_weighted_null_acceleration"], 0, atol=1e-13
    )
    # A real right-null coefficient carries arbitrary unseen-axis response.
    np.testing.assert_allclose(
        saved["query_null_acceleration"][0], np.arange(1, 7) + 1 / 3, atol=1e-13
    )


def test_independent_curvature_remains_when_affine_column_space_is_smaller():
    x = np.array([-1, 0, 1, 2], float)
    affine = np.c_[np.ones(4), x]
    quadratic = x[:, None] ** 2
    summary, saved = analyze_design(
        affine,
        quadratic,
        np.array([[1, 3.0]]),
        np.array([[9.0]]),
        np.full(4, 0.25),
        np.zeros((2, 6)),
        np.ones((1, 6)),
    )
    assert summary["affine"]["rank"] == 2
    assert summary["conditional_quadratic"]["independent_rows"] == 2
    assert summary["conditional_quadratic"]["rank"] == 1
    assert (
        summary["conditional_quadratic"]["query_support"][0][
            "right_null_energy_fraction"
        ]
        < 1e-25
    )
    np.testing.assert_allclose(
        saved["conditional_unsupported_acceleration"], 0, atol=1e-13
    )


def test_zero_and_unsupported_query_directions_do_not_claim_support():
    decomposition = svd_design(np.array([[1.0, 0, 0], [2.0, 0, 0]]))
    values, _projected, null = support_rows(
        np.array([[0.0, 0, 0], [0.0, 1, 0]]), decomposition
    )
    assert values[0]["rowspace_energy_fraction"] is None
    assert values[0]["right_null_energy_fraction"] is None
    assert values[1]["rowspace_energy_fraction"] == 0
    assert values[1]["right_null_energy_fraction"] == 1
    np.testing.assert_array_equal(null[1], [0, 1, 0])
    empty = svd_design(np.empty((0, 3)))
    assert empty["rank"] == 0 and empty["row_basis"].shape == (0, 3)
    expected = 3 * np.finfo(np.float64).eps * np.sqrt(5)
    assert decomposition["tolerance"] == pytest.approx(expected)


def test_command_swap_preserves_each_context_and_never_scores_counterfactual_truth(
    monkeypatch,
):
    from pathlib import Path
    from types import SimpleNamespace

    import diagnose_online_support as diagnostic

    captured = {}
    for origin in (156, 157):
        context = dict(
            past_states=np.full((2, 15), float(origin)),
            past_inputs=np.full((1, 3), float(-origin)),
            command=np.array([origin, 0, 0], float),
            truth=np.full(15, float(origin)),
        )
        diagonal = np.full(15, 2.0 * origin)
        captured[origin] = (
            SimpleNamespace(fingerprint="same", params={}, norms={}),
            {},
            context,
            {"factor_1_prediction": diagonal, "factor_64_prediction": diagonal},
        )
    calls, truth_calls = [], []
    monkeypatch.setattr(
        diagnostic, "load_capture", lambda path: captured[int(path.name)]
    )

    def integrate(model, past, past_inputs, command, factor):
        origin = int(past[-1, 0])
        assert np.all(past_inputs == -origin)
        calls.append((origin, int(command[0]), factor))
        return dict(
            prediction=np.full(15, origin + command[0]),
            states=past[-1].reshape(1, 1, 15),
            filtered=np.zeros((1, 1, 3)),
            history=np.zeros((1, 15)),
            hidden=np.zeros(1),
        )

    def metrics(prediction, truth):
        truth_calls.append(int(truth[0]))
        return {"error": float(np.linalg.norm(prediction - truth))}

    monkeypatch.setattr(diagnostic, "integrate_stages", integrate)
    monkeypatch.setattr(
        diagnostic,
        "_stage_fields",
        lambda *args: {"head_contributions": np.zeros((1, 6, 6))},
    )
    monkeypatch.setattr(diagnostic, "physical_metrics", metrics)
    summary, _ = diagnostic.command_swaps(Path("unused"))
    assert len(calls) == 8 and truth_calls == [156, 156, 157, 157]
    for entry in summary.values():
        for resolution in entry["resolutions"].values():
            assert ("factual_truth_errors" in resolution) == entry["factual"]
