"""Synthetic saved-evidence checks; no benchmark collection or optimizer fits."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from test_excited_public_matching import prepared_pair

from glassbox import learner
from glassbox._learner_arrays import array_fingerprint
from glassbox.experimental import command_excitation_fit as excitation
from glassbox.experimental import excited_bilinear_matching as matching
from glassbox.experimental import excited_public_matching as shared
from glassbox.experimental import state_input_model as bilinear
from glassbox.experimental import state_quadratic_model as quadratic


@pytest.fixture(scope="module")
def prepared():
    """Truthful initial means; synthetic trace metadata only tests validation."""
    public, quadratic_model, roles = prepared_pair()
    train, development = public._train, public._development
    sequence = bilinear.initialize_candidate(
        train.batch, ridge=0.01 * 384 * 5, delay_steps=2
    )
    counts = {
        "baseline": sum(v.size for v in public._model.params.values()),
        "candidate": sum(v.size for v in sequence.params.values()),
        "quadratic": sum(v.size for v in quadratic_model._model.params.values()),
    }
    report = copy.deepcopy(public.report)
    report.pop("previous_revision")
    report["recipe"] = copy.deepcopy(bilinear.RECIPE)
    report["optimization"].update(
        mechanism=bilinear.EXPERIMENT,
        minibatch_seed=10000,
        parameter_count=counts["candidate"],
    )
    report["matching"] = dict(
        baseline_fingerprint=public.fingerprint(),
        training_windows_fingerprint=bilinear._window_fingerprint(train),
        development_windows_fingerprint=bilinear._window_fingerprint(development),
        baseline_norms_fingerprint=array_fingerprint({}, public._model.norms),
        error_scale_fingerprint=array_fingerprint(
            {}, {"error_scale": np.asarray(report["optimization"]["error_scale"])}
        ),
        shared_initialization_fingerprint=array_fingerprint(
            {},
            {
                key: sequence.params[key]
                for key in ("w1", "b1", "w2", "memory", "memory_bias")
            },
        ),
        minibatch_seed=10000,
        optimizer_steps=1000,
        batch_size=64,
        baseline_parameter_count=counts["baseline"],
        candidate_parameter_count=counts["candidate"],
        equal_compute=False,
    )
    candidate = bilinear.CandidateDynamics(
        sequence, train, development, public.contract, report, np.ones((5, 2))
    )
    return public, candidate, quadratic_model, roles, counts


@pytest.fixture
def trio(prepared, monkeypatch):
    public, candidate, quadratic_model, roles, counts = copy.deepcopy(prepared)
    old_protocol = shared._protocol()
    old_protocol["planned_automatic_roles"] = {"synthetic": roles}
    old_protocol["imported_calibration"]["excitation_seal_sha256"] = {
        "synthetic": "a" * 64
    }
    monkeypatch.setattr(shared, "_protocol", lambda: copy.deepcopy(old_protocol))
    protocol = matching._protocol()
    protocol["planned_automatic_roles"] = {"synthetic": roles}
    protocol["imported_calibration"]["excitation_seal_sha256"] = {"synthetic": "a" * 64}
    protocol["fitting"]["mechanism"]["counts"] = {"synthetic": counts}
    protocol["trusted_comparator_revisions"] = {
        "synthetic": {
            "baseline": {"revision_fingerprint": public.fingerprint()},
            "quadratic": {"revision_fingerprint": quadratic_model.fingerprint()},
        }
    }
    monkeypatch.setattr(matching, "_protocol", lambda: copy.deepcopy(protocol))
    return public, candidate, quadratic_model


def test_matching_replays_stably_without_fit_solve_rollout_or_mutation(
    trio, monkeypatch, tmp_path
):
    def forbidden(*args, **kwargs):
        raise AssertionError("matching may not fit, initialize, solve or roll out")

    for module, names in (
        (learner, ("fit", "_train", "initialize_sequence_model")),
        (bilinear, ("fit_candidate", "fit_candidate_sequence", "initialize_candidate")),
        (quadratic, ("fit_candidate_sequence", "initialize_candidate")),
        (excitation, ("fit_excited",)),
        (np.linalg, ("solve", "lstsq")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    for model in trio:
        monkeypatch.setattr(type(model._model), "rollout", forbidden)
    before = [model.fingerprint() for model in trio]
    expected = matching.check_matched_models(*trio)
    assert [model.fingerprint() for model in trio] == before
    assert expected["window_counts"] == {"training": 384, "development": 256}
    assert expected["exact_same_data"] and expected["exact_same_numeric_objective"]
    assert expected["selected_initial_shared_parameters_checked"] is True
    assert expected["equal_flops_claimed"] is False
    assert expected["equal_end_to_end_compute_claimed"] is False
    assert len(expected["source_sha256"]) == 9
    restored = []
    for index, model in enumerate(trio):
        path = tmp_path / f"model-{index}.npz"
        model.save(path)
        restored.append(type(model).load(path))
    assert matching.check_matched_models(*restored) == expected
    assert json.loads(json.dumps(expected, allow_nan=False)) == expected


@pytest.mark.parametrize("role", ["_train", "_development"])
@pytest.mark.parametrize(
    "field", ["past_states", "past_inputs", "future_inputs", "future_states"]
)
def test_every_changed_bilinear_cached_array_is_rejected(trio, role, field):
    candidate = trio[1]
    windows = getattr(candidate, role)
    values = getattr(windows.batch, field).copy()
    values[0, 0, 0] += 0.01
    setattr(
        candidate,
        role,
        replace(windows, batch=replace(windows.batch, **{field: values})),
    )
    with pytest.raises(ValueError, match=r"bilinear .* cache"):
        matching.check_matched_models(*trio)


@pytest.mark.parametrize("change", ["key", "origin", "dt", "count"])
def test_cache_metadata_is_not_replaced_by_matching_values(trio, change):
    candidate = trio[1]
    windows = candidate._development
    if change == "key":
        candidate._development = replace(windows, keys=tuple(reversed(windows.keys)))
    elif change == "origin":
        candidate._development = replace(
            windows, source_origins=(999, *windows.source_origins[1:])
        )
    elif change == "dt":
        candidate._development = replace(
            windows, batch=replace(windows.batch, dt_s=0.1)
        )
    else:
        candidate._development = learner._subset(windows, np.arange(255))
    with pytest.raises(ValueError, match="bilinear development cache"):
        matching.check_matched_models(*trio)


@pytest.mark.parametrize("norm", [*shared.BASE_NORMS, "interaction_scale"])
def test_all_shared_and_product_norms_must_derive_from_exact_train(trio, norm):
    candidate = trio[1]
    norms = {key: value.copy() for key, value in candidate._model.norms.items()}
    norms[norm] += 0.01
    candidate._model = replace(candidate._model, norms=norms)
    with pytest.raises(ValueError, match="normalization"):
        matching.check_matched_models(*trio)


@pytest.mark.parametrize(
    "key,value",
    [
        ("baseline_fingerprint", "0" * 64),
        ("training_windows_fingerprint", "0" * 64),
        ("development_windows_fingerprint", "0" * 64),
        ("baseline_norms_fingerprint", "0" * 64),
        ("error_scale_fingerprint", "0" * 64),
        ("shared_initialization_fingerprint", "0" * 64),
        ("minibatch_seed", 10001),
        ("optimizer_steps", 999),
        ("batch_size", 63),
        ("baseline_parameter_count", 1),
        ("candidate_parameter_count", 1),
        ("equal_compute", True),
    ],
)
def test_matching_metadata_cannot_claim_a_different_reference_or_budget(
    trio, key, value
):
    trio[1]._report["matching"][key] = value
    with pytest.raises(ValueError, match="matching report"):
        matching.check_matched_models(*trio)


@pytest.mark.parametrize(
    "change",
    [
        "contract",
        "scale",
        "recipe",
        "steps",
        "schedule",
        "selection",
        "tie",
        "nonfinite",
        "seed",
        "mechanism",
        "shape",
        "update",
        "coverage",
        "timing",
        "initial_draw",
    ],
)
def test_bilinear_semantic_mismatches_rejected(trio, change):
    candidate = trio[1]
    report = candidate._report
    optimization = report["optimization"]
    if change == "contract":
        candidate._contract["state_channels"][0] = "other:m"
    elif change == "scale":
        optimization["error_scale"][0][0] += 0.1
    elif change == "recipe":
        report["recipe"]["memory"] = 4
    elif change == "steps":
        optimization["steps"] = 999
    elif change == "schedule":
        optimization["trace"][1]["step"] = 99
    elif change == "selection":
        optimization["selected_step"] = 100
    elif change == "tie":
        optimization["trace"][1]["validation_rollout_mse"] = 1.0
        optimization["selected_step"] = 100
    elif change == "nonfinite":
        optimization["trace"][1]["training_batch_mse"] = float("nan")
    elif change == "seed":
        optimization["minibatch_seed"] = 10001
    elif change == "mechanism":
        optimization["mechanism"] = "autonomous-state-quadratic-v1"
    elif change == "shape":
        candidate._model.params["autonomous"] = np.zeros((3, 2))
    elif change == "update":
        report["previous_revision"] = "0" * 64
    elif change == "coverage":
        report["training"]["window_count"] = 383
    elif change == "timing":
        report["delay_steps"] = 1
    else:
        candidate._model.params["w1"] = candidate._model.params["w1"] + 0.01
    with pytest.raises(ValueError):
        matching.check_matched_models(*trio)


@pytest.mark.parametrize("change", ["roles", "content", "seal", "revision"])
def test_reference_provenance_cannot_be_relabelled(trio, change):
    public, _, quadratic_model = trio
    if change == "roles":
        quadratic_model._report["provenance"]["roles"]["training"].reverse()
    elif change == "content":
        public._seen[next(iter(public._seen))] = "0" * 64
    elif change == "seal":
        quadratic_model._report["provenance"]["data_seal"] = "0" * 64
    else:
        public._report["untrusted_extra"] = True
    with pytest.raises(ValueError):
        matching.check_matched_models(*trio)


def test_fixed_parameter_counts_are_checked(trio, monkeypatch):
    protocol = matching._protocol()
    protocol["fitting"]["mechanism"]["counts"]["synthetic"]["candidate"] += 1
    monkeypatch.setattr(matching, "_protocol", lambda: protocol)
    with pytest.raises(ValueError, match="parameter counts"):
        matching.check_matched_models(*trio)


@pytest.mark.parametrize("index", [0, 2])
def test_public_or_quadratic_cannot_stand_in_for_bilinear(trio, index):
    with pytest.raises(TypeError, match="bilinear"):
        matching.check_matched_models(trio[0], trio[index], trio[2])


@pytest.mark.parametrize(
    "suffix", ["excited_public_matching.py", "excited-public-architecture-v1.json"]
)
def test_inherited_witness_source_and_protocol_pins_are_verified(
    trio, monkeypatch, suffix
):
    digest = matching._digest
    monkeypatch.setattr(
        matching,
        "_digest",
        lambda path: "0" * 64 if str(path).endswith(suffix) else digest(path),
    )
    with pytest.raises(ValueError, match="frozen"):
        matching.check_matched_models(*trio)


def test_changed_fitting_source_is_rejected(trio, monkeypatch):
    digest = shared._digest
    monkeypatch.setattr(
        shared,
        "_digest",
        lambda path: (
            "0" * 64 if str(path).endswith("state_input_model.py") else digest(path)
        ),
    )
    with pytest.raises(ValueError, match="frozen fitting source"):
        matching.check_matched_models(*trio)


def test_ablation_protocol_is_byte_pinned(monkeypatch):
    monkeypatch.setattr(matching, "_digest", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="frozen bilinear ablation protocol"):
        matching._protocol()
