"""Expanded-data initialization/acceptance observation on small toy caches only."""

import copy
import json
import sys
from dataclasses import replace

import jax
import numpy as np
import pytest
import test_full_cache_gradient_checkpoints as prior
from test_fit_checkpoints import (
    PROVENANCE,
    SETTINGS,
    TOY_RECIPE,
    assert_model_equal,
    toy_scale,
    windows,
)

from glassbox._learner_arrays import array_fingerprint, load_arrays, save_arrays
from glassbox.experimental import expanded_training_cache_checkpoints as checkpoints
from glassbox.experimental import expanded_training_cache_model as model

SETTINGS_NEW = {**SETTINGS, "batch_size": 48, "ridge": 4.0}
RECIPE_NEW = {**TOY_RECIPE, "training_windows": 48, "batch_size": 48}


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def source_map():
    names = (
        "docs/harness/expanded-training-cache-v1.json",
        "src/glassbox/experimental/expanded_training_cache_model.py",
        "src/glassbox/experimental/expanded_training_cache_checkpoints.py",
        "tests/test_expanded_training_cache_model.py",
        "tests/test_expanded_training_cache_checkpoints.py",
    )
    return {
        **prior.source_map(),
        **{p: checkpoints._sha(checkpoints.ROOT / p) for p in names},
    }


def subset_anchor(path):
    _, arrays = load_arrays(path)
    subset = {k: arrays["param_" + k] for k in checkpoints.SUBSET}
    return dict(
        imported_arm="fullcache384",
        source_arm="candidate",
        imported_relative_path="toy/fullcache384/checkpoints/weighted/anchored/trajectory/step-0000.npz",
        source_relative_path="toy/candidate/checkpoints/weighted/anchored/trajectory/step-0000.npz",
        snapshot_sha256=checkpoints._sha(path),
        **{
            f"source_{name}_manifest_sha256": checkpoints._sha(
                path.parents[level] / "manifest.json"
            )
            for level, name in enumerate(
                (
                    "trajectory",
                    "anchored_checkpoint",
                    "weighted_checkpoint",
                    "checkpoint",
                )
            )
        },
        subset_parameter_names=list(checkpoints.SUBSET),
        subset_arrays=checkpoints._subset_description(subset),
        subset_array_fingerprint=array_fingerprint({}, subset),
    )


def expand(train):
    parts = [train] + [windows(i) for i in (4, 5, 6)]
    batch = replace(
        train.batch,
        **{
            key: np.concatenate([getattr(p.batch, key) for p in parts])
            for key in checkpoints._base.ARRAYS
        },
    )
    keys = tuple(
        replace(key, recording_id=f"added-{i}-{key.recording_id}") if i else key
        for i, part in enumerate(parts)
        for key in part.keys
    )
    return replace(
        train,
        batch=batch,
        keys=keys,
        source_origins=tuple(v for p in parts for v in p.source_origins),
    )


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    historical = prior.fitted.__wrapped__(tmp_path_factory)
    old = tmp_path_factory.mktemp("external") / "checkpoints"
    prior.save_fitted(old, historical)
    snapshot = old / "weighted/anchored/trajectory/step-0000.npz"
    train, dev = expand(historical["train"]), historical["dev"]
    context = checkpoints.initialization_context(
        train=train,
        development=dev,
        recipe=RECIPE_NEW,
        preparation={"toy": True},
        snapshot_path=snapshot,
        trusted_subset=subset_anchor(snapshot),
        source_bundle_sha256="a" * 64,
    )
    with jax.enable_x64(True):
        plain, plain_report = model.fit_candidate_sequence(
            train.batch, dev.batch, **SETTINGS_NEW, error_scale=toy_scale(train)
        )
        solve = np.linalg.solve
        draws = np.random.default_rng
        calls = {"solve": 0, "rng": 0}

        def counted_solve(*args, **kwargs):
            calls["solve"] += 1
            return solve(*args, **kwargs)

        def counted_rng(*args, **kwargs):
            calls["rng"] += 1
            return draws(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(np.linalg, "solve", counted_solve)
            patch.setattr(np.random, "default_rng", counted_rng)
            before = sys.gettrace()
            with checkpoints.capture_fit(
                "candidate",
                PROVENANCE,
                source_sha256=source_map(),
                expected_steps=(0, 1, 2),
            ) as captured:
                observed, report = model.fit_candidate_sequence(
                    train.batch, dev.batch, **SETTINGS_NEW, error_scale=toy_scale(train)
                )
            assert sys.gettrace() is before
        assert calls == dict(solve=2, rng=1)
    return dict(
        historical=historical,
        train=train,
        dev=dev,
        context=context,
        plain=plain,
        plain_report=plain_report,
        observed=observed,
        report=report,
        captured=captured,
    )


def save_fitted(path, fitted, *, captured=None):
    captured = fitted["captured"] if captured is None else captured
    report = fitted["report"]
    evidence = checkpoints.save_checkpoints(
        path,
        captured,
        train=fitted["train"],
        development=fitted["dev"],
        recipe=RECIPE_NEW,
        initialization_context=fitted["context"],
        selected_step=report["selected_step"],
        selected_model=fitted["observed"],
    )
    kwargs = dict(
        train=fitted["train"],
        development=fitted["dev"],
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        initialization_context=fitted["context"],
        selected_model=fitted["observed"],
        expectation=dict(
            arm="candidate",
            recipe=RECIPE_NEW,
            expected_steps=[0, 1, 2],
            completed=True,
            failure=None,
            selected_step=report["selected_step"],
            trace=report["trace"],
            unweighted_trace=report["unweighted_development_trace"],
            objective=report["objective"],
            safeguard=report["safeguard"],
            gradient=report["gradient"],
            status="complete",
        ),
    )
    return evidence, kwargs


def reseal(path):
    for file in sorted(
        path.rglob("manifest.json"), key=lambda p: len(p.parts), reverse=True
    ):
        manifest = json.loads(file.read_text())
        manifest["files"] = {
            str(p.relative_to(file.parent)): checkpoints._sha(p)
            for p in file.parent.rglob("*")
            if p.is_file() and p != file
        }
        for key, relative in [
            ("trajectory_manifest_sha256", "trajectory/manifest.json"),
            ("anchored_manifest_sha256", "anchored/manifest.json"),
            ("weighted_manifest_sha256", "weighted/manifest.json"),
        ]:
            if key in manifest:
                manifest[key] = checkpoints._sha(file.parent / relative)
        file.write_text(json.dumps(manifest))
    return checkpoints._sha(path / "manifest.json")


def test_actual_parity_current_initial_changes_and_external_subset(fitted):
    assert_model_equal(fitted["plain"], fitted["observed"])
    assert fitted["plain_report"] == fitted["report"]
    state = fitted["captured"].initializer_return
    assert (state["entered"], state["returned"], state["exceptional_unwinds"]) == (
        1,
        1,
        0,
    )
    assert all(not v.flags.writeable for v in state["arrays"].values())
    old = fitted["historical"]["captured"].snapshots[0]
    for key in checkpoints.SUBSET:
        np.testing.assert_array_equal(
            state["arrays"]["param_" + key], old["params"][key]
        )
    assert not np.array_equal(state["arrays"]["param_linear"], old["params"]["linear"])
    assert not np.array_equal(
        state["arrays"]["norm_state_mean"], old["norms"]["state_mean"]
    )
    assert not np.array_equal(
        fitted["captured"].weight_observation["arrays"]["channel_weights"],
        fitted["historical"]["captured"].weight_observation["arrays"][
            "channel_weights"
        ],
    )


def test_complete_save_replay_and_partitions_no_solve_gradient_or_rng(
    tmp_path, fitted, monkeypatch
):
    path = tmp_path / "new"
    evidence, kwargs = save_fitted(path, fitted)

    def forbidden(*args, **kwargs):
        pytest.fail("replay executed solve, gradient or RNG")

    for obj, name in [
        (np.linalg, "solve"),
        (np.random, "default_rng"),
        (jax, "grad"),
        (jax, "value_and_grad"),
    ]:
        monkeypatch.setattr(obj, name, forbidden)
    result = checkpoints.replay_checkpoints(path, **kwargs)
    assert (
        result["exact"]
        and result["checkpoints"] == result["training_partition_checkpoints"] == 3
    )
    assert result["within_fit_initial_identity"]["scope"] == "within_current_fit"
    assert result["gradient"]["known_gradient_window_visits"] == 96
    assert result["initializer_return_links"]["installed_return_link"]
    assert evidence["initializer_return_available"]
    partitions = json.loads((path / "training-partitions.json").read_text())
    assert [
        partitions["partitions"][k]["windows"]
        for k in ("full", "original_prefix", "added")
    ] == [48, 12, 36]
    for row in partitions["checkpoints"]:
        p = row["partitions"]
        for key in ("original_loss", "reconstructed_selection_loss"):
            assert np.isclose(
                p["full"]["diagnostics"][key],
                0.25 * p["original_prefix"]["diagnostics"][key]
                + 0.75 * p["added"]["diagnostics"][key],
                rtol=1e-12,
            )
    assert "balanced_initial_witness" not in (path / "attempts.json").read_text()
    assert (
        "within_fit_initial_identity" in (path / "weighted/manifest.json").read_text()
    )
    assert checkpoints.previous._model() is prior.model


@pytest.mark.parametrize(
    "change",
    (
        "returned_parameter",
        "returned_norm",
        "removed_return",
        "weight",
        "ridge",
        "partition",
        "indices",
        "dtype",
        "proposal",
        "acceptance",
        "gradient_count",
        "initial_loss",
        "missing_rhs",
        "false_initial_identity",
    ),
)
def test_resealed_semantic_evidence_rejected(tmp_path, fitted, change):
    path = tmp_path / "new"
    _, kwargs = save_fitted(path, fitted)
    if change in ("returned_parameter", "returned_norm", "removed_return"):
        file = path / "initializer-return.npz"
        meta, arrays = load_arrays(file)
        if change == "removed_return":
            meta["witness"]["returned"] = 0
        else:
            arrays[
                "param_w1" if change == "returned_parameter" else "norm_state_mean"
            ].flat[0] += 0.01
        save_arrays(file, meta, arrays)
    elif change == "weight":
        file = path / "weighted/objective-witness.npz"
        meta, arrays = load_arrays(file)
        arrays["channel_weights"][0] *= 1.01
        save_arrays(file, meta, arrays)
    elif change in ("ridge", "missing_rhs"):
        file = path / "weighted/anchored/initializer-witness.npz"
        meta, arrays = load_arrays(file)
        if change == "ridge":
            meta["witness"]["settings"]["ridge"] /= 4
        else:
            del meta["witness"]["captured_fingerprints"]["centered_rhs"]
        save_arrays(file, meta, arrays)
    elif change == "partition":
        file = path / "training-partitions.json"
        meta = json.loads(file.read_text())
        meta["checkpoints"][0]["partitions"]["added"]["diagnostics"][
            "original_loss"
        ] += 0.1
        file.write_text(json.dumps(meta))
    elif change == "false_initial_identity":
        file = path / "weighted/manifest.json"
        meta = json.loads(file.read_text())
        meta["within_fit_initial_identity"]["scope"] = "same_as_imported"
        file.write_text(json.dumps(meta))
    else:
        file = path / "attempts.json"
        meta = json.loads(file.read_text())
        row = meta["attempts"][0]
        if change == "indices":
            row["minibatch_indices"][0] = 1
        elif change == "dtype":
            row["gradient_indices_dtype"] = "<i4"
        elif change == "proposal":
            f = path / "proposals.npz"
            m, a = load_arrays(f)
            a["proposal_bias"][0, 0] += 0.1
            save_arrays(f, m, a)
        elif change == "acceptance":
            row["accepted_scale"] = 0.3
        elif change == "gradient_count":
            meta["gradient"]["known_gradient_window_visits"] = 24
        else:
            meta["initial"]["loss"]["value"] *= 2
        file.write_text(json.dumps(meta))
    kwargs["expected_sha256"] = reseal(path)
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.replay_checkpoints(path, **kwargs)


def test_wrong_external_source_and_missing_subset_rejected(fitted):
    for field, value in [
        ("source_arm", "balanced"),
        ("subset_parameter_names", list(checkpoints.SUBSET[:-1])),
        ("snapshot_sha256", "0" * 64),
    ]:
        context = copy.deepcopy(fitted["context"])
        context["external_subset"][field] = value
        with pytest.raises(checkpoints.CheckpointError):
            checkpoints._context(context, fitted["train"], fitted["dev"], RECIPE_NEW)


def capture_failure(fitted, stage):
    jit, solve = jax.jit, np.linalg.solve
    solve_calls = []

    def failed_solve(*args, **kwargs):
        solve_calls.append(1)
        if len(solve_calls) == 2:
            raise np.linalg.LinAlgError("injected joint solve failure")
        return solve(*args, **kwargs)

    def failed_scan(*args, **kwargs):
        raise model.CandidateFitError("injected initial forecast failure")

    def compiled(function, *args, **kwargs):
        result = jit(function, *args, **kwargs)
        if (stage == "development" and function.__name__ == "evaluate") or (
            stage == "proposal" and function.__name__ == "update"
        ):

            def fail(*args):
                raise model.CandidateFitError("injected " + stage + " failure")

            return fail
        return result

    with pytest.MonkeyPatch.context() as patch:
        if stage == "initializer":
            patch.setattr(np.linalg, "solve", failed_solve)
        elif stage == "weighting":
            patch.setattr(jax.lax, "scan", failed_scan)
        elif stage in ("development", "proposal"):
            patch.setattr(jax, "jit", compiled)
        else:

            class Clock:
                calls = 0

                @classmethod
                def perf_counter(cls):
                    cls.calls += 1
                    return 0.0 if cls.calls <= 5 else 7201.0

            patch.setattr(model._engine, "time", Clock)
        previous = sys.gettrace()
        with pytest.raises(
            (ValueError, np.linalg.LinAlgError), match=r"injected|fit_time_limit"
        ):
            with checkpoints.capture_fit(
                "candidate",
                PROVENANCE,
                source_sha256=source_map(),
                expected_steps=(0, 1, 2),
            ) as captured:
                model.fit_candidate_sequence(
                    fitted["train"].batch,
                    fitted["dev"].batch,
                    **SETTINGS_NEW,
                    error_scale=toy_scale(fitted["train"]),
                )
        assert sys.gettrace() is previous
    return captured


def save_failure(path, fitted, captured):
    failure = captured.failure["message"]
    evidence = checkpoints.save_checkpoints(
        path,
        captured,
        train=fitted["train"],
        development=fitted["dev"],
        recipe=RECIPE_NEW,
        initialization_context=fitted["context"],
        failure=failure,
    )
    return dict(
        train=fitted["train"],
        development=fitted["dev"],
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        initialization_context=fitted["context"],
        expectation=dict(
            arm="candidate",
            recipe=RECIPE_NEW,
            expected_steps=[0, 1, 2],
            completed=False,
            failure=failure,
            selected_step=None,
            trace=None,
            unweighted_trace=None,
            objective=None,
            safeguard=None,
            gradient=None,
            status="fit_failure",
        ),
    )


@pytest.mark.parametrize(
    "stage", ("initializer", "weighting", "development", "proposal", "final_timeout")
)
def test_actual_initializer_and_optimizer_failure_prefixes(tmp_path, fitted, stage):
    captured = capture_failure(fitted, stage)
    path = tmp_path / "failed"
    kwargs = save_failure(path, fitted, captured)
    result = checkpoints.replay_checkpoints(path, **kwargs)
    state = captured.initializer_return
    assert state["entered"] == 1
    assert state["returned"] == int(stage != "initializer")
    assert state["exceptional_unwinds"] == int(stage == "initializer")
    assert result["initializer_return_available"] == (stage != "initializer")
    expected_checkpoints = 3 if stage == "final_timeout" else int(stage == "proposal")
    assert result["checkpoints"] == expected_checkpoints
    assert result["weighting_available"] == (stage not in ("initializer", "weighting"))
    assert result["gradient"]["incomplete_gradient_work_unknown"] == (
        stage == "proposal"
    )


@pytest.fixture(scope="module")
def before_weight_failure(fitted):
    with jax.enable_x64(True):
        return capture_failure(fitted, "weighting")


@pytest.mark.parametrize(
    "change",
    ("extra_array", "bias_shape", "linear_shape", "norm_shape", "external_subset"),
)
def test_full_return_roster_shapes_and_external_anchor_before_weights(
    tmp_path, fitted, before_weight_failure, change
):
    path = tmp_path / "failed"
    kwargs = save_failure(path, fitted, before_weight_failure)
    file = path / "initializer-return.npz"
    meta, arrays = load_arrays(file)
    if change == "extra_array":
        arrays["unobserved"] = np.zeros(1)
    elif change == "bias_shape":
        arrays["param_bias"] = arrays["param_bias"][None, :]
    elif change == "linear_shape":
        arrays["param_linear"] = arrays["param_linear"].reshape(-1)
    elif change == "norm_shape":
        arrays["norm_state_scale"] = arrays["norm_state_scale"][None, :]
    else:
        arrays["param_w1"][0, 0] += 0.01
    save_arrays(file, meta, arrays)
    kwargs["expected_sha256"] = reseal(path)
    with pytest.raises(checkpoints.CheckpointError, match=r"roster|shape|subset"):
        checkpoints.check_checkpoint_links(path, **kwargs)


def test_prepared_but_unentered_fitter_retains_empty_witness(tmp_path, fitted):
    with pytest.raises(ValueError, match="injected before fitter"):
        with checkpoints.capture_fit(
            "candidate",
            PROVENANCE,
            source_sha256=source_map(),
            expected_steps=(0, 1, 2),
        ) as captured:
            raise ValueError("injected before fitter")
    path = tmp_path / "unentered"
    kwargs = save_failure(path, fitted, captured)
    result = checkpoints.replay_checkpoints(path, **kwargs)
    assert not result["initializer_return_available"]
    assert result["checkpoints"] == result["gradient"]["attempts_started"] == 0


def test_historical_modules_and_numerical_code_unchanged(fitted):
    assert (
        model.fit_candidate_sequence.__code__
        == prior.model.fit_candidate_sequence.__code__
    )
    assert checkpoints.previous._model() is prior.model
    assert (
        checkpoints.weighted._model()
        is checkpoints.previous.previous.inherited._model()
    )
    assert checkpoints.previous.FORMAT == "glassbox-full-cache-gradient-checkpoints-v1"
    assert (
        checkpoints.previous.PROTOCOL_SHA256
        == "fe318d07998e0f05385e421a930f172453641f30e020724550d8811d7757976d"
    )
