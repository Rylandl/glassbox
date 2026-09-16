"""M2 acceptance must gate on capability, ordinary guards, and frozen cases."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import check_generic_memory

    return check_generic_memory


def fixture():
    ordinary = ["a", "b"]
    manifest = dict(
        id="unit",
        required_families=ordinary,
        capability_families=["delayed_nonlinear", "hidden_input_delay"],
        stress_families=["noise"],
        data_seeds=[1],
        extra_regression_cases=[],
        hidden_input_delay=dict(data_seeds=[7], regimes=["matched"]),
        compute=dict(
            max_training_window_gradient_evaluations=100, max_development_passes=12
        ),
        error_caps={
            **{
                f: dict(matched=1.0, shifted=1.0)
                for f in ["a", "b", "noise", "delayed_nonlinear"]
            },
            "hidden_input_delay": dict(matched=1.0),
        },
        capability=dict(
            delayed_nonlinear_maximum_ratio=0.5,
            hidden_input_delay_maximum_ratio=0.5,
            paired_probe_first_step_rmse_maximum=0.05,
        ),
        guards=dict(ordinary_maximum_ratio=1.05),
        primary=dict(floor=0.005, maximum_ratio=0.95),
    )

    def metric(value):
        return dict(horizon_scaled_rmse=[value] * 5, overall_scaled_rmse=value)

    def row(family, seed, candidate, regimes=("matched", "shifted")):
        return dict(
            name=f"{family}-{seed}",
            family=family,
            data_seed=seed,
            extra=False,
            status="complete",
            correctness_passed=True,
            optimization=dict(
                resource_usage=dict(
                    training_window_gradient_evaluations=100, development_passes=11
                )
            ),
            regimes={
                r: dict(incumbent=metric(0.2), candidate=metric(candidate))
                for r in regimes
            },
        )

    rows = [
        row("a", 1, 0.2),
        row("b", 1, 0.2),
        row("noise", 1, 0.21),
        row("delayed_nonlinear", 1, 0.08),
        row("hidden_input_delay", 7, 0.05, regimes=("matched",)),
    ]
    rows[-1]["probe"] = dict(
        incumbent_paired_rmse=[0.2, 0.16, 0.128, 0.1024, 0.08192],
        candidate_paired_rmse=[0.02, 0.01, 0.01, 0.01, 0.01],
    )
    return manifest, rows


def test_capability_gates_and_ordinary_guard_pass_together(runner):
    manifest, rows = fixture()
    answer = runner.decide(manifest, rows)
    assert answer["accepted"] and answer["decision"] == "replace"
    assert answer["ordinary_guard"]["ratio"] == pytest.approx(1.0)
    assert answer["per_family_ratios"]["delayed_nonlinear"] == pytest.approx(0.4)
    assert answer["per_family_ratios"]["hidden_input_delay"] == pytest.approx(0.25)
    assert answer["primary_ratio"] == pytest.approx(
        np.exp(np.mean([0, 0, np.log(0.4), np.log(0.25)]))
    )
    assert answer["paired_probe_first_step_rmse"] == {"7": 0.02}
    json.dumps(answer, allow_nan=False)


@pytest.mark.parametrize(
    "breach",
    ["probe", "delayed_ratio", "witness_ratio", "ordinary_guard", "cap", "compute"],
)
def test_capability_gain_cannot_hide_a_failed_gate(runner, breach):
    manifest, rows = fixture()
    if breach == "probe":
        rows[-1]["probe"]["candidate_paired_rmse"][0] = 0.06
    if breach == "delayed_ratio":
        for metrics in rows[3]["regimes"].values():
            metrics["candidate"]["overall_scaled_rmse"] = 0.12
    if breach == "witness_ratio":
        rows[-1]["regimes"]["matched"]["candidate"]["overall_scaled_rmse"] = 0.11
    if breach == "ordinary_guard":
        for row in rows[:2]:
            for metrics in row["regimes"].values():
                metrics["candidate"]["overall_scaled_rmse"] = 0.2 * 1.06
    if breach == "cap":
        rows[0]["regimes"]["shifted"]["candidate"]["horizon_scaled_rmse"][3] = 1.01
    if breach == "compute":
        rows[1]["optimization"]["resource_usage"]["development_passes"] = 13
    answer = runner.decide(manifest, rows)
    assert not answer["accepted"] and answer["decision"] == "retain"
    if breach in ("probe", "delayed_ratio", "witness_ratio"):
        assert answer["capability_gate_breaches"]
    elif breach == "ordinary_guard":
        assert not answer["ordinary_guard"]["passed"]
        assert answer["primary_ratio"] < 0.95
    else:
        assert answer["hard_gate_breaches"]
    json.dumps(answer, allow_nan=False)


def test_missing_duplicate_or_wrong_regime_cases_are_rejected(runner):
    manifest, rows = fixture()
    for invalid in [rows[:-1], rows + [copy.deepcopy(rows[0])]]:
        with pytest.raises(ValueError, match="frozen cases"):
            runner.decide(manifest, invalid)
    manifest, rows = fixture()
    rows[-1]["regimes"]["shifted"] = rows[-1]["regimes"]["matched"]
    with pytest.raises(ValueError, match="evaluation regimes"):
        runner.decide(manifest, rows)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, None])
def test_invalid_probe_or_score_fails_closed_and_stays_serializable(runner, value):
    manifest, rows = fixture()
    if value is None:
        del rows[-1]["probe"]
    else:
        rows[-1]["probe"]["candidate_paired_rmse"][0] = value
        rows[3]["regimes"]["matched"]["candidate"]["overall_scaled_rmse"] = value
    answer = runner.decide(manifest, rows)
    assert not answer["accepted"]
    json.dumps(answer, allow_nan=False)


@pytest.mark.parametrize("mode", ["fit", "verify"])
def test_manifest_changes_cannot_create_a_new_acceptance_contract(
    runner, tmp_path, mode
):
    path = tmp_path / "manifest.json"
    path.write_text('{"id": "generic-fit-m2-v1"}\n')
    with pytest.raises(ValueError, match="manifest digest"):
        if mode == "fit":
            runner.main(SimpleNamespace(manifest=path, output=tmp_path / "output"))
        else:
            runner.verify_run(tmp_path, path)
    assert not (tmp_path / "output").exists()


def test_numpy_replay_matches_the_filter_rollout_without_a_fit(runner):
    import jax

    from glassbox.experimental.sequence_model import (
        initialize_sequence_model,
        sequence_windows,
    )

    rng = np.random.default_rng(100)
    b = sequence_windows(
        rng.normal(size=(61, 2)),
        rng.normal(size=(60, 3)),
        np.arange(10, 50),
        history_steps=10,
        horizon_steps=5,
        dt_s=0.05,
    )
    model = initialize_sequence_model(
        b, kind="filter_mlp", width=4, memory=3, delay_steps=2
    )
    model.params["linear"][-3:] = 0.2
    model.params["w2"][:] = 0.05
    with jax.enable_x64(True):
        expected = np.asarray(
            model.rollout(b.past_states, b.past_inputs, b.future_inputs)
        )
    actual = runner.recurrence_memory(
        model, b.past_states, b.past_inputs, b.future_inputs
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
    # The memory must matter for this check to be meaningful.
    blind = initialize_sequence_model(
        b, kind="filter_mlp", width=4, memory=3, delay_steps=2
    )
    blind.params["w2"][:] = 0.05
    assert not np.allclose(
        blind.rollout(b.past_states, b.past_inputs, b.future_inputs), expected
    )


def test_context_probe_matches_the_qualification_probe(runner, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from experiment_model_qualification import memory_probe

    px, pu, uf, target = runner.context_probe(10, 5)
    _, _, short_uf, short_target = memory_probe()
    np.testing.assert_allclose(target, short_target, atol=1e-12)
    np.testing.assert_array_equal(uf, short_uf)
    np.testing.assert_array_equal(px[0], px[1])
    assert px.shape == (2, 11, 1) and pu.shape == (2, 10, 1)
    assert np.flatnonzero(pu[0] != pu[1]).tolist() == [7]
    np.testing.assert_array_equal(pu[:, 7, 0], [-1.0, 1.0])
    # Nothing has happened yet at the origin: identical contexts, different futures.
    np.testing.assert_array_equal(px, 0)
    assert runner.lengths(
        dict(context_s=0.5, delay_s=0.1, horizon_s=0.25), 0.05
    ) == dict(context=10, delay=2, horizon=5)
