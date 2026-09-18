"""Synthetic horizon, startup, telemetry and denominator tests; no simulator."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_jsbsim_onboarding_audit import evidence as onboarding_evidence

from glassbox.experimental import jsbsim_onboarding_audit as old
from glassbox.experimental.jsbsim_excitation_audit import (
    EXTRA_ARRAYS,
    common_tape,
    requested_telemetry,
    summarize,
    validate_arm,
)


@pytest.fixture
def evidence():
    reference_case, reference_arrays, inherited, entry = onboarding_evidence.__wrapped__()
    reference_case["computed"] = old.validate_case(reference_case, reference_arrays, inherited, entry)
    protocol = json.loads((Path(__file__).resolve().parents[1] / "docs/harness/jsbsim-excitation-v1.json").read_text())
    case = deepcopy(reference_case)
    case.pop("computed")
    case["arm"] = "as_shipped"
    arrays = {}
    tape = common_tape(protocol, inherited, entry, reference_case, reference_arrays)
    bounds = np.asarray(reference_case["runs"]["parent"]["bounds"])
    requested = requested_telemetry(protocol, 1)
    names = ["fcs/elevator-pos-rad", "position/h-agl-ft", "propulsion/engine/set-running", "propulsion/engine/thrust-lbs"]
    names = [x for x in requested if x in names]
    for label in case["runs"]:
        command = tape.copy()
        branch = next((row for row in case["branches"] if row["label"] == label), None)
        if branch is not None and branch["command_index"] is not None:
            col = branch["command_index"]
            bound = bounds[col, int(branch["direction"] == "upper")]
            command[20:, col] += .1 * (bound - command[20:, col])
        native = np.zeros((121, 16))
        native[:, 0] = 20 + np.r_[0, np.cumsum(command[:, 0])]
        native[:, 4] = np.r_[0, np.cumsum(command[:, 1])]
        native[:, 15] = 500 + np.r_[0, np.cumsum(command[:, 3])]
        native[:, 6:15] = reference_arrays["parent__native_observations"][0, 6:15]
        observed = old._mapped(native)
        telem = np.column_stack([np.r_[0, command[:, 1]], native[:, 15], np.zeros(121), np.r_[0, command[:, 3]] * 10])
        data = dict(commands=command, observations=observed, native_observations=native,
                    time_s=np.arange(121, dtype=np.float64) * .05,
                    immediate_readbacks=np.repeat(command[:, None, :], 6, axis=1),
                    end_readbacks=np.repeat(command[:, None, :], 6, axis=1),
                    native_attempted=np.ones((120, 6)), native_completed=np.ones((120, 6)),
                    failed_native_observation=np.empty((0, 16)), failed_observation=np.empty((0, 16)),
                    failed_time_s=np.empty(0), telemetry=telem, telemetry_valid=np.ones_like(telem),
                    failed_telemetry=np.empty((0, len(names))), failed_telemetry_valid=np.empty((0, len(names))))
        metadata = case["runs"][label]
        metadata.update(completed_intervals=120,
                        telemetry=dict(names=names.copy(), missing=[x for x in requested if x not in names],
                                       errors=[{} for _ in range(121)], failed_errors=[]),
                        startup=dict(arm="as_shipped", native_run_ic_attempted_count=1,
                                     native_run_ic_returned_count=1, native_run_ic_result=True, native_run_ic_error=None,
                                     attempted_count=0, returned_count=0, error=None, phase="ready",
                                     snapshot_errors=dict(pre={}, post={})))
        for side in ("pre", "post"):
            data.update({
                f"startup_{side}_telemetry": telem[:1].copy(),
                f"startup_{side}_telemetry_valid": np.ones((1, len(names))),
                f"startup_{side}_native_observation": native[:1].copy(),
                f"startup_{side}_observation": observed[:1].copy(),
                f"startup_{side}_observation_valid": np.ones((1, 16)),
                f"startup_{side}_commands": np.asarray([metadata["diagnostics"]["initial_commands"]]),
                f"startup_{side}_commands_valid": np.ones((1, 4)),
                f"startup_{side}_time_s": np.zeros(1), f"startup_{side}_time_valid": np.ones(1),
            })
        arrays.update({f"{label}__{key}": value for key, value in data.items()})
    return case, arrays, protocol, inherited, entry, reference_case, reference_arrays


def _validate(evidence):
    result = validate_arm(*evidence)
    json.dumps(result, allow_nan=False)
    return result


def _bootstrap(evidence):
    evidence[0]["arm"] = "engine_bootstrap"
    for meta in evidence[0]["runs"].values():
        meta["startup"].update(arm="engine_bootstrap", attempted_count=1, returned_count=1)


def _fail(evidence, label, completed):
    case, arrays = evidence[:2]
    case["runs"][label].update(status="recording_failure", stage="observation", error="nonfinite",
                                completed_intervals=completed, failed_interval=completed, failed_substep=5)
    for name in ("observations", "native_observations", "time_s", "telemetry", "telemetry_valid"):
        arrays[f"{label}__{name}"] = arrays[f"{label}__{name}"][:completed + 1].copy()
    for name in ("immediate_readbacks", "end_readbacks", "native_attempted", "native_completed"):
        arrays[f"{label}__{name}"] = arrays[f"{label}__{name}"][:completed + 1].copy()
    arrays[f"{label}__failed_native_observation"] = np.full((1, 16), np.nan)
    arrays[f"{label}__failed_observation"] = np.full((1, 16), np.nan)
    arrays[f"{label}__failed_time_s"] = np.array([(completed + 1) * .05])
    case["runs"][label]["telemetry"]["errors"] = case["runs"][label]["telemetry"]["errors"][:completed + 1]


def test_full_physical_horizons_and_exact_historical_reference(evidence):
    result = _validate(evidence)
    assert result["outcome"] == "completed"
    assert result["run_counts"] == dict(planned=11, attempted=11, completed=11, unattempted=0)
    assert result["historical_prefixes"]["parent"] == {"compared_steps": 40, "equal": True}
    assert result["historical_prefixes"]["c000_lower"]["compared_steps"] == 25
    for h in (5, 20, 40, 100):
        row = result["horizon_results"][str(h)]
        assert (row["available"], row["cumulative_active"], row["endpoint_active"], row["weak"]) == (4, 3, 3, 1)
    assert result["commands"][0]["first_detection_any"]["time_s"] == .05
    assert result["commands"][2]["first_detection_any"] == dict(time_s=None, observed_through_s=5., right_censored=False)


def test_late_parent_failure_preserves_short_horizons(evidence):
    # The old simulator failed earlier in this synthetic reference as well, so
    # compare only its available prefix rather than fabricate historical parity.
    _bootstrap(evidence)
    for label in ("parent", "replay", "factual"):
        _fail(evidence, label, 29)
    result = _validate(evidence)
    assert result["outcome"] == "parent_failure"
    assert result["horizon_results"]["5"]["available"] == 4
    assert result["horizon_results"]["20"]["available"] == 0
    assert result["commands"][2]["first_detection_any"] == dict(time_s=None, observed_through_s=.45, right_censored=True)


def test_late_one_sided_failure_affects_only_one_command(evidence):
    _fail(evidence, "c000_lower", 45)
    result = _validate(evidence)
    assert result["outcome"] == "branch_failure"
    assert result["horizon_results"]["20"]["available"] == 4
    assert result["horizon_results"]["40"]["available"] == 3
    assert result["commands"][0]["observed_through_steps"] == 25


def test_failed_core_getter_can_have_no_failed_telemetry_row(evidence):
    _fail(evidence, "c000_lower", 30)
    assert len(evidence[1]["c000_lower__failed_observation"]) == 1
    assert len(evidence[1]["c000_lower__failed_telemetry"]) == 0
    assert _validate(evidence)["horizon_results"]["5"]["available"] == 4


def test_factual_divergence_gates_only_later_core_horizons(evidence):
    _bootstrap(evidence)
    evidence[1]["factual__native_observations"][40, 0] += 1
    evidence[1]["factual__observations"] = old._mapped(evidence[1]["factual__native_observations"])
    result = _validate(evidence)
    assert result["outcome"] == "replay_failure"
    assert result["horizon_results"]["5"]["available"] == 4
    assert result["horizon_results"]["20"]["available"] == 0


def test_core_response_does_not_depend_on_telemetry_availability_or_parity(evidence):
    arrays = evidence[1]
    arrays["c000_lower__telemetry"][25, 0] = np.nan
    arrays["c000_lower__telemetry_valid"][25, 0] = 0
    path = evidence[0]["runs"]["c000_lower"]["telemetry"]["names"][0]
    evidence[0]["runs"]["c000_lower"]["telemetry"]["errors"][25][path] = "RuntimeError: missing"
    arrays["replay__telemetry"][0, 0] += 1
    result = _validate(evidence)
    assert result["horizon_results"]["100"]["available"] == 4
    assert result["telemetry_parity"]["replay"]["equal_where_valid"] is False
    assert result["commands"][0]["telemetry"]["lower"][path]["signed_delta"][4] is None


def test_cumulative_and_endpoint_activity_are_distinct(evidence):
    _bootstrap(evidence)
    for direction in ("lower", "upper"):
        label = f"c002_{direction}"
        evidence[1][label + "__native_observations"][21, 5] = .1
        evidence[1][label + "__observations"][21, 5] = .1
    result = _validate(evidence)
    activity = result["horizon_results"]["5"]["activity"][2]
    assert activity["cumulative_active"] is True
    assert activity["endpoint_active"] is False


@pytest.mark.parametrize("mutation,match", [
    (lambda c, a: c["branches"][1].update(command_index=2), "branch roster"),
    (lambda c, a: a["parent__commands"].__setitem__((80, 0), .7), "common intended tape"),
    (lambda c, a: a["c000_lower__commands"].__setitem__((80, 0), .7), "common intended tape"),
    (lambda c, a: a["parent__telemetry_valid"].__setitem__((5, 0), 0), "finite/valid"),
    (lambda c, a: c["runs"]["parent"]["telemetry"]["names"].reverse(), "path order"),
    (lambda c, a: c["runs"]["parent"]["startup"].update(attempted_count=1), "as-shipped"),
    (lambda c, a: a["parent__startup_pre_observation"].__setitem__((0, 0), 0), "conversion"),
])
def test_rehashed_structural_corruption_rejected(evidence, mutation, match):
    mutation(evidence[0], evidence[1])
    with pytest.raises(ValueError, match=match):
        _validate(evidence)


def test_historical_parity_is_required_even_with_coherent_new_trajectories(evidence):
    for label in evidence[0]["runs"]:
        evidence[1][label + "__native_observations"][:, 0] += 1
        evidence[1][label + "__observations"] = old._mapped(evidence[1][label + "__native_observations"])
    with pytest.raises(ValueError, match="historical as-shipped"):
        _validate(evidence)


def test_timeout_retains_known_roster_but_disables_all_physical_horizons(evidence):
    case, arrays = evidence[:2]
    labels = list(case["runs"])
    label = "c001_lower"
    for later in labels[labels.index(label) + 1:]:
        case["runs"][later] = {"status": "unattempted"}
        for key in (*old.ARRAY_NAMES, *EXTRA_ARRAYS):
            del arrays[f"{later}__{key}"]
    case["runs"][label] = {"status": "running", "stage": "construct"}
    for key in (*old.ARRAY_NAMES, *EXTRA_ARRAYS):
        del arrays[f"{label}__{key}"]
    case.update(execution_status="timeout", timeout_s=180, returncode=None)
    result = _validate(evidence)
    assert result["outcome"] == "timeout"
    assert result["run_counts"]["planned"] == 11
    assert all(row["available"] == 0 for row in result["horizon_results"].values())


def test_timeout_before_first_snapshot_retains_reference_contract(evidence):
    case = evidence[0]
    for label in case["runs"]:
        case["runs"][label] = {"status": "unattempted"}
    case["runs"]["parent"] = {"status": "running", "stage": "construct"}
    evidence[1].clear()
    case.update(execution_status="timeout", timeout_s=180, returncode=None)
    result = _validate(evidence)
    assert result["command_contract_source"] == "reference"
    assert result["run_counts"] == dict(planned=11, attempted=1, completed=0, unattempted=10)
    assert result["horizon_results"]["5"]["unavailable_known"] == 4


def test_summary_uses_matched_denominators_and_fixed_cohort(evidence):
    baseline = deepcopy(evidence)
    baseline[0]["computed"] = _validate(baseline)
    candidate = deepcopy(evidence)
    _bootstrap(candidate)
    for direction in ("lower", "upper"):
        label = f"c002_{direction}"
        candidate[1][label + "__native_observations"][21:, 5] = .1
        candidate[1][label + "__observations"][21:, 5] = .1
    _fail(candidate, "c000_lower", 45)
    candidate[0]["computed"] = _validate(candidate)
    inventory = {"entries": [evidence[4]], "directories": ["example", "LM"]}
    result = summarize([baseline[0], candidate[0]], inventory, evidence[2])
    matched = result["all_candidates"]["paired_horizons"]
    assert matched["20"]["cumulative_active"]["matched_available"] == 4
    assert matched["20"]["cumulative_active"]["gain"] == 1
    assert matched["40"]["cumulative_active"]["matched_available"] == 3
    assert matched["40"]["cumulative_active"]["transitions"]["active_to_unavailable"] == 1
    assert result["old_completed_cohort"] == [evidence[4]["id"]]
    assert result["directories"][1]["candidates"] == []
    assert result["all_candidates"]["arms"]["as_shipped"]["run_counts"]["known_plan"]["attempted"] == 11


def test_summary_rejects_missing_arm_and_changed_cohort(evidence):
    case = evidence[0]
    case["computed"] = _validate(evidence)
    inventory = {"entries": [evidence[4]], "directories": ["example"]}
    with pytest.raises(ValueError, match="inventory accounting"):
        summarize([case], inventory, evidence[2])
    candidate = deepcopy(case)
    candidate["arm"] = "engine_bootstrap"
    candidate["computed"]["old_completed_cohort"] = False
    with pytest.raises(ValueError, match="cohort differs"):
        summarize([case, candidate], inventory, evidence[2])


def test_successful_ic_cannot_strip_diagnostic_roster(evidence):
    metadata = evidence[0]["runs"]["parent"]
    metadata["telemetry"].update(names=[], missing=[])
    for name in ("telemetry", "telemetry_valid", "failed_telemetry", "failed_telemetry_valid",
                 "startup_pre_telemetry", "startup_pre_telemetry_valid", "startup_post_telemetry", "startup_post_telemetry_valid"):
        evidence[1]["parent__" + name] = evidence[1]["parent__" + name][:, :0]
    with pytest.raises(ValueError, match="path accounting"):
        _validate(evidence)


def test_native_ic_result_requires_boolean_evidence(evidence):
    evidence[0]["runs"]["parent"]["startup"]["native_run_ic_result"] = 1
    with pytest.raises(ValueError, match="native IC result"):
        _validate(evidence)
