"""Prospective pure-data/worker-boundary tests; no model fit or simulator."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox.experimental import public_mean_physical_evaluation as subject
from glassbox.experimental import state_input_decision as paired
from glassbox.experimental.two_simulator_metrics import GROUPS, score_forecast

ROOT = Path(__file__).resolve().parents[1]


def qualification():
    return json.loads(
        (ROOT / "docs/harness/public-mean-qualification-v1.json").read_text()
    )


def test_finite_prefix_inference_does_not_use_truth_to_truncate():
    calls = []
    values = dict(
        past_states=np.zeros((3, 15)),
        past_inputs=np.zeros((2, 2)),
        future_inputs=np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 0.0]]),
        valid=np.array([False, False, False]),
    )

    def predict(x, u, f):
        calls.append(f.copy())
        return np.ones((len(f), 15), dtype=np.float32)

    result = subject.predict_query(
        predict, {"history_eligible": True}, values, dtype=np.float32
    )
    assert result.dtype == np.float32 and len(calls) == 1
    np.testing.assert_array_equal(calls[0], values["future_inputs"][:2])
    assert np.isfinite(result[:2]).all() and np.isnan(result[2]).all()
    result = subject.predict_query(
        predict, {"history_eligible": False}, values, dtype=np.float32
    )
    assert np.isnan(result).all() and len(calls) == 1


def test_prediction_precision_contract_is_not_silently_cast():
    values = dict(
        past_states=np.zeros((2, 15)),
        past_inputs=np.zeros((1, 1)),
        future_inputs=np.zeros((1, 1)),
    )
    with pytest.raises(subject.EvaluationError) as error:
        subject.predict_query(
            lambda *a: np.zeros((1, 15), dtype=np.float64),
            {"history_eligible": True},
            values,
            dtype=np.float32,
        )
    assert error.value.check_id == "prediction_contract"


@pytest.mark.parametrize("change", ["truth_mask", "command"])
def test_truth_reconstruction_rejects_query_mutation_against_raw_parent(
    tmp_path, monkeypatch, change
):
    from glassbox.experimental import two_simulator_flight as flight

    p = {"generation": {"toy": {"dt_s": 0.25}}}
    states = np.zeros((4, 13))
    states[:, 2], states[:, 6] = 1.0, 1.0
    parent = {
        "states": states,
        "full_states": states.copy(),
        "commands": np.full((3, 1), 0.5),
    }
    path = tmp_path / "parents/one.npz"
    path.parent.mkdir()
    np.savez_compressed(path, **parent)
    record = {"id": "one", "prefix": "parents/one", "validity": flight.validity(parent)}
    query = {
        "parent": "one",
        "id": "factual",
        "origin": 2,
        "history_eligible": True,
        "kind": "factual",
        "path": "query.npz",
    }
    config = {"spec": {"channels": [{"kind": "control"}]}, "bounds": [[0.0, 1.0]]}
    observed = np.concatenate(
        (np.zeros((4, 6)), np.tile(np.eye(3).reshape(1, 9), (4, 1))), axis=1
    )
    monkeypatch.setattr(flight, "OBSERVE", lambda value: observed.copy())
    monkeypatch.setattr(subject, "_planned", lambda *a: ([record], [query], config))
    values = {
        "past_states": observed[:3],
        "past_inputs": parent["commands"][:2],
        "future_inputs": parent["commands"][2:].copy(),
        "factual_inputs": parent["commands"][2:].copy(),
        "target": observed[3:].copy(),
        "factual_target": observed[3:].copy(),
        "valid": np.ones(1, bool),
        "command_delta": np.zeros((1, 1)),
    }
    np.savez_compressed(tmp_path / "query.npz", **values)
    assert subject.reconstruct_truth(tmp_path, "toy", p)[
        "exact_input_truth_mask_arrays"
    ]
    if change == "truth_mask":
        values["valid"][:] = False
    else:
        values["future_inputs"][0, 0] += 0.01
    np.savez_compressed(tmp_path / "query.npz", **values)
    with pytest.raises(subject.EvaluationError) as error:
        subject.reconstruct_truth(tmp_path, "toy", p)
    assert error.value.check_id == "truth_roster_and_masks"


def test_calibration_reconstructs_rank_and_rejects_coherent_envelope_change():
    import jax

    target = np.arange(256.0).reshape(256, 1, 1) * np.ones((1, 1, 15))
    batch = SimpleNamespace(
        past_states=np.zeros((256, 2, 15)),
        past_inputs=np.zeros((256, 1, 1)),
        future_inputs=np.zeros((256, 1, 1)),
        future_states=target,
    )
    calls = []

    def rollout(*args):
        assert jax.config.x64_enabled
        calls.append("actual-selected-mean")
        return np.zeros((256, 1, 15))

    half_width = np.full((1, 15), 231.0)
    report = dict(
        optimization={"selected_step": 1000},
        envelope=dict(
            nominal_coverage=0.9,
            method="split_conformal_absolute_error",
            calibrated_on="development",
            calibration_windows=256,
            quantile_rank=232,
            units="physical, per horizon step and per channel, aligned with predict",
            half_width=half_width.tolist(),
        ),
    )
    model = SimpleNamespace(
        _development=SimpleNamespace(batch=batch),
        _model=SimpleNamespace(rollout=rollout),
        report=report,
        envelope=lambda: half_width.copy(),
    )
    before = bool(jax.config.x64_enabled)
    witness, evidence = subject.calibration_evidence(model)
    assert evidence["quantile_rank"] == 232 and calls == ["actual-selected-mean"]
    np.testing.assert_array_equal(witness["half_width"], half_width)
    assert bool(jax.config.x64_enabled) == before
    half_width[0, 0] += 0.001
    report["envelope"]["half_width"] = half_width.tolist()
    with pytest.raises(subject.EvaluationError) as error:
        subject.calibration_evidence(model)
    assert error.value.check_id == "calibration_envelope_reconstruction"


def toy_rows(*, local_loss=False):
    p = dict(
        id="toy",
        cells={},
        generation={},
        recordings=[],
        evaluation={"horizons_s": [0.25]},
        decision={
            "aggregation": {
                "simulator_weights": {"crazyflow": 0.5, "cascade": 0.5},
                "scope_weights": {"primary": 0.5, "shifted": 0.5},
                "horizons_s": [0.25],
                "groups": list(GROUPS),
                "floors": {k: 0.01 for k in GROUPS},
            }
        },
    )
    rows, queries = {}, {}
    for sim in ("crazyflow", "cascade"):
        p["generation"][sim] = {"dt_s": 0.25}
        p["cells"][sim] = []
        rows[sim], queries[sim] = [], []
        for scope in ("primary", "shifted"):
            cell, parent = scope + "-cell", sim + "/" + scope + "-parent"
            p["cells"][sim].append(dict(id=cell, group=scope))
            p["recordings"].append(
                dict(id=parent, role="test", simulator=sim, cell=cell)
            )
            for kind, sign in (("factual", None), ("response", -1), ("response", 1)):
                q = dict(
                    id=kind + str(sign),
                    simulator=sim,
                    parent=parent,
                    scope=scope,
                    cell=cell,
                    origin=2,
                    kind=kind,
                )
                if sign is not None:
                    q.update(sign=sign, channel=0)
                queries[sim].append(q)
                for arm in subject.ARMS:
                    error = 1.0 if arm == "public_v3" else 0.8
                    if local_loss and arm != "public_v3":
                        error = 1.2 if (sim, scope) == ("crazyflow", "primary") else 0.1
                    for score in score_forecast(
                        np.full((1, 15), error),
                        np.zeros((1, 15)),
                        np.ones(1, bool),
                        dt_s=0.25,
                        horizons_s=[0.25],
                        rotation_geometry=False,
                    ):
                        row = dict(
                            score,
                            simulator=sim,
                            parent=parent,
                            scope=scope,
                            cell=cell,
                            origin=2,
                            kind=kind,
                            query=q["id"],
                            arm=arm,
                            channel=q.get("channel"),
                            sign=sign,
                        )
                        if kind == "response":
                            row["pair_nonweak"] = True
                        rows[sim].append(row)
    return rows, queries, p


def test_real_operators_keep_broad_gain_without_v3_local_veto():
    rows, queries, p = toy_rows(local_loss=True)
    before = copy.deepcopy((rows, p))
    result = subject.reduce_decision(rows, p, qualification(), queries=queries)
    assert result["fresh_default32_physical_comparisons_pass"]
    v3 = result["comparisons"]["public_v3"]
    assert set(v3["checks"]) == {
        "matching_planned_queries_and_truth",
        "finite_eligible_predictions",
        "factual_aggregate",
        "response_aggregate",
    }
    assert any(row["ratio"] > 1.05 for row in v3["tail_checks"])
    assert all("pass" not in row for row in v3["tail_checks"])
    assert result["shared_bootstrap_parent_draws_verified"]
    assert result["public_mean_adopted"] is False
    assert (rows, p) == before


def test_finite_large_float64_diagnostic_residuals_do_not_veto_public32():
    rows, queries, p = toy_rows()
    for values in rows.values():
        for row in values:
            if row["arm"] == "public_v4_float64":
                row.update(mse=1e12, finite_subset_mse=1e12)
    result = subject.reduce_decision(rows, p, qualification(), queries=queries)
    assert result["all_arm_eligible_predictions_finite"] is True
    assert result["fresh_default32_physical_comparisons_pass"] is True
    assert result["float64_is_diagnostic"] is True


@pytest.mark.parametrize(
    "change",
    [
        "omitted_query_all_arms",
        "omitted_arm",
        "duplicated",
        "mask",
        "diagnostic_nonfinite",
    ],
)
def test_reducer_cannot_gain_from_omitted_or_failed_slots(change):
    rows, queries, p = toy_rows()
    values = rows["crazyflow"]
    if change == "omitted_query_all_arms":
        rows["crazyflow"] = [
            r
            for r in values
            if not (
                r["parent"] == queries["crazyflow"][0]["parent"]
                and r["query"] == queries["crazyflow"][0]["id"]
            )
        ]
    elif change == "omitted_arm":
        rows["crazyflow"] = [r for r in values if r["arm"] != "hold"]
    elif change == "duplicated":
        values[-1] = copy.deepcopy(values[0])
    elif change == "mask":
        next(r for r in values if r["arm"] == "hold")["truth_eligible"] = False
    else:
        row = next(r for r in values if r["arm"] == "public_v4_float64")
        row.update(prediction_finite=False, mse=None)
        result = subject.reduce_decision(rows, p, qualification(), queries=queries)
        assert result["all_arm_eligible_predictions_finite"] is False
        assert result["fresh_default32_physical_comparisons_pass"] is False
        return
    with pytest.raises(subject.EvaluationError) as error:
        subject.reduce_decision(rows, p, qualification(), queries=queries)
    assert error.value.check_id == "truth_roster_and_masks"


@pytest.mark.parametrize(
    "reference,kind,limit",
    [
        ("public_v3", "factual", 0.9),
        ("public_v3", "response", 0.9),
        ("research", "factual", 1.01),
        ("research", "response", 1.01),
    ],
)
def test_aggregate_thresholds_are_inclusive_and_next_float_fails(
    monkeypatch, reference, kind, limit
):
    rows, queries, p = toy_rows()
    original = paired.reduce
    # Keep real cohort/mean/bootstrap operators, replace only the terminal scalar
    # to probe an exact boundary independent of transcendental roundoff.
    override = {"value": limit}
    calls = []

    def boundary(*args, **kwargs):
        result = original(*args, **kwargs)
        target = "public_v3" if len(calls) % 2 == 0 else "research"
        calls.append(target)
        if target == reference:
            result["weighted_geometric_mean_ratios"][kind] = override["value"]
        return result

    monkeypatch.setattr(paired, "reduce", boundary)
    result = subject.reduce_decision(rows, p, qualification(), queries=queries)
    assert result["comparisons"][reference]["checks"][kind + "_aggregate"]
    override["value"] = np.nextafter(limit, np.inf)
    result = subject.reduce_decision(rows, p, qualification(), queries=queries)
    assert not result["comparisons"][reference]["checks"][kind + "_aggregate"]


def test_research_local_guards_remain_required(monkeypatch):
    rows, queries, p = toy_rows()
    original = paired.reduce

    def worse_tail(*args, **kwargs):
        result = original(*args, **kwargs)
        result["checks"]["primary_parent_tail_regressions"] = False
        return result

    monkeypatch.setattr(paired, "reduce", worse_tail)
    result = subject.reduce_decision(rows, p, qualification(), queries=queries)
    assert result["comparisons"]["public_v3"]["residual_criteria_pass"]
    assert not result["comparisons"]["research"]["residual_criteria_pass"]
    assert not result["fresh_default32_physical_comparisons_pass"]


def test_promotion_requires_all_named_boolean_qualifications():
    q = qualification()
    gates = {name: True for name in q["public_mean_promotion"]["all_required"]}
    physical = {"fresh_default32_physical_comparisons_pass": True}
    assert subject.promotion(physical, gates, q)["public_mean_adopted"]
    gates["independent_dart_consumer_pass"] = False
    assert not subject.promotion(physical, gates, q)["public_mean_adopted"]
    gates["fresh_default32_physical_comparisons_pass"] = False
    with pytest.raises(subject.EvaluationError, match="physical qualification mirror"):
        subject.promotion(physical, gates, q)
    del gates["fresh_default32_physical_comparisons_pass"]
    with pytest.raises(subject.EvaluationError, match="qualification roster"):
        subject.promotion(physical, gates, q)


def _binding():
    return dict(
        format="glassbox-public-mean-implementation-v1",
        protocol_sha256=subject.PROTOCOL_SHA256,
        implementation_commit="old",
        public_root="/old",
        public_source_sha256={
            "src/glassbox/learner.py": "learner",
            "src/glassbox/_sequence_model.py": "model",
        },
        interpreter="/python",
        interpreter_sha256="python",
        runtime={"jax": "pinned"},
        machine="arm64",
        oracle_commit=subject.ORACLE_COMMIT,
        oracle_source_sha256={"old.py": "old"},
        consumer_source_sha256={"consumer.py": "consumer"},
    )


def test_prior_stage_binding_requires_exact_old_source_subset(tmp_path):
    old = _binding()
    path = tmp_path / "old-binding.json"
    subject.write(path, old)
    current = copy.deepcopy(old)
    current["implementation_commit"] = "new"
    current["public_source_sha256"][subject.RELATIVE] = "new-evaluator"
    assert subject.stage_binding(path, subject.digest(path), current) == old
    current["public_source_sha256"]["src/glassbox/_sequence_model.py"] = "changed"
    with pytest.raises(subject.EvaluationError, match="exact source superset"):
        subject.stage_binding(path, subject.digest(path), current)
    with pytest.raises(subject.EvaluationError, match="external stage binding"):
        subject.stage_binding(path, "0" * 64, current)


def test_inventory_includes_nested_seals_and_rejects_links(tmp_path):
    subject.write(tmp_path / "seal.json", {})
    subject.write(tmp_path / "nested/seal.json", {"nested": True})
    assert set(subject.inventory(tmp_path)) == {"nested/seal.json"}
    (tmp_path / "link").symlink_to(tmp_path / "nested/seal.json")
    with pytest.raises(subject.EvaluationError):
        subject.inventory(tmp_path)


def test_launcher_retains_timeout_and_does_not_retry(tmp_path, monkeypatch):
    binding = dict(public_root=tmp_path, oracle_root=tmp_path, interpreter="python")
    calls = []

    class Process:
        pid = 123

        def wait(self, timeout=None):
            calls.append(timeout)
            if timeout is not None:
                raise subject.subprocess.TimeoutExpired("worker", timeout)
            return -9

    def launch(*args, **kwargs):
        assert kwargs["env"].get("JAX_ENABLE_X64") is None
        return Process()

    monkeypatch.setattr(subject.subprocess, "Popen", launch)
    monkeypatch.setattr(subject.os, "killpg", lambda *args: None)
    with pytest.raises(subject.EvaluationError, match="no retry"):
        subject._launch({"role": "public"}, tmp_path / "launch", binding)
    result = subject.read(tmp_path / "launch/execution.json")
    assert result["hard_timeout"] and result["status"] == "failed"
    assert calls == [14400, None]


def test_saved_scoring_reconstructs_response_in_float64_and_complete_roster(
    tmp_path, monkeypatch
):
    p = dict(
        generation={"toy": {"dt_s": 0.25}},
        evaluation={
            "horizons_s": [0.25],
            "responses": {"weak_endpoint_thresholds": {k: 0.01 for k in GROUPS}},
        },
    )
    queries = [
        dict(
            parent="parent",
            id="factual",
            kind="factual",
            origin=2,
            scope="primary",
            cell="cell",
            simulator="toy",
            path="factual.npz",
            history_eligible=True,
        )
    ]
    queries += [
        dict(
            queries[0],
            id="response" + str(sign),
            kind="response",
            path="response" + str(sign) + ".npz",
            sign=sign,
            channel=0,
        )
        for sign in (-1, 1)
    ]
    data, evaluation = tmp_path / "data", tmp_path / "evaluation"
    data.mkdir()
    for role, arms in subject.ROLES.items():
        (evaluation / role).mkdir(parents=True)
        np.savez_compressed(
            evaluation / role / "envelopes.npz",
            **{arm: np.ones((1, 15)) for arm in arms},
        )
    for query in queries:
        values = dict(
            past_states=np.zeros((2, 15)),
            past_inputs=np.zeros((1, 1)),
            future_inputs=np.ones((1, 1)),
            factual_inputs=np.zeros((1, 1)),
            target=np.full((1, 15), query.get("sign", 0)),
            factual_target=np.zeros((1, 15)),
            valid=np.ones(1, bool),
        )
        np.savez_compressed(data / query["path"], **values)
        for role, arms in subject.ROLES.items():
            values = {}
            for arm in arms:
                dtype = np.float32 if arm == "public_v4" else np.float64
                values[arm] = np.full((1, 15), query.get("sign", 0), dtype=dtype)
                if query["kind"] == "response":
                    values[arm + "_factual"] = np.zeros((1, 15), dtype=dtype)
            path = evaluation / role / subject.query_path(query)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **values)
    monkeypatch.setattr(subject, "_planned", lambda *args: ([], queries, {}))
    result = subject.score(data, evaluation, "toy", p)
    assert len(result["rows"]) == 3 * 5 * 3 * 2
    assert all(row["mse"] == 0 for row in result["rows"] if row["arm"] != "hold")
    assert all(
        row["coverage"] == 1
        for row in result["rows"]
        if row["kind"] == "factual" and row["arm"] != "hold"
    )
    assert any(
        row["zero_prediction"] for row in result["directions"] if row["arm"] == "hold"
    )
    path = evaluation / "public" / subject.query_path(queries[0])
    path.unlink()
    with pytest.raises(FileNotFoundError):
        subject.score(data, evaluation, "toy", p)
