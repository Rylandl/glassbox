"""Independent saved-scalar and physical endpoint arithmetic; no learner imports."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import verify_online_v8 as audit


def proposal(losses=(11.0, 10.0, 9.0), predicted=1.0, **updates):
    result = dict(
        current_loss=10.0,
        conditioning_finite=True,
        direction_finite=True,
        trust_shrink=0.75,
        trials=[
            dict(alpha=alpha, loss=loss, predicted_reduction=predicted, finite=True)
            for alpha, loss in zip(audit.ALPHAS, losses)
        ],
        selected_alpha=audit.ALPHAS[len(losses) - 1],
        trial_evaluations=len(losses),
    )
    result.update(updates)
    return result


def report(count, objective_calls, accepted, damping, last):
    return dict(
        observations=count,
        optimizer_steps=count,
        gradient_calls=count,
        objective_calls=objective_calls,
        accepted_proposals=accepted,
        cg_iterations=16 * count,
        curvature_calls=16 * count,
        conditioning_calls=count,
        damping=damping,
        last_proposal=last,
    )


def test_first_acceptable_shortening_and_nonfinite_trial_rescue():
    result = proposal()
    result["trials"][0].update(loss=None, finite=False)
    assert audit.proposal_decision(result) == dict(
        accepted=True, gain=1.0, trials=3, selected_alpha=0.25
    )
    before = report(0, 0, 0, 1.0, None)
    after = report(1, 4, 1, 0.5, result)
    assert audit.verify_transition(before, after)["trials"] == 3


@pytest.mark.parametrize(
    "gain,expected", [(0.1, 4.0), (0.249, 4.0), (0.25, 1.0), (0.75, 1.0), (0.751, 0.5)]
)
def test_original_gain_and_damping_thresholds(gain, expected):
    # Use exactly representable improvement; predicted chooses desired ratio.
    p = proposal((9.0,), predicted=1.0 / gain)
    decision = audit.verify_transition(
        report(0, 0, 0, 1.0, None), report(1, 2, 1, expected, p)
    )
    assert decision["accepted"]


@pytest.mark.parametrize("field", ["conditioning_finite", "direction_finite"])
def test_invalid_header_disables_entire_ladder_and_rolls_damping_once(field):
    p = proposal((9.0,) * 5, selected_alpha=0.0, **{field: False})
    if field == "direction_finite":
        p.update(current_loss=None, trust_shrink=None)
    assert not audit.verify_transition(
        report(0, 0, 0, 1.0, None), report(1, 6, 0, 4.0, p)
    )["accepted"]


def test_rejection_all_trials_can_log_finite_losses_with_false_trial_validity():
    p = proposal((9.0,) * 5, selected_alpha=0.0)
    for trial in p["trials"]:
        trial["finite"] = False
    assert not audit.proposal_decision(p)["accepted"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["trials"][0].update(loss=9.0),
        lambda p: p["trials"][1].update(alpha=0.25),
        lambda p: p.update(selected_alpha=0.5),
        lambda p: p.update(trial_evaluations=4),
        lambda p: p.update(selected_alpha=0.0),
        lambda p: p.update(direction_finite=False),
        lambda p: p["trials"][0].update(loss=float("nan"), finite=False),
        lambda p: p["trials"][0].update(loss=None),
        lambda p: p.update(current_loss=None),
        lambda p: p.update(trust_shrink=2.0),
        lambda p: p.update(trial_evaluations=True),
        lambda p: p.update(current_loss=-1.0),
        lambda p: p["trials"][0].update(loss=-1.0),
    ],
)
def test_proposal_tampering_is_rejected(mutate):
    p = proposal()
    mutate(p)
    with pytest.raises(AssertionError):
        audit.proposal_decision(p)


@pytest.mark.parametrize(
    "field,value",
    [
        ("objective_calls", 3),
        ("accepted_proposals", 0),
        ("damping", 1.0),
        ("cg_iterations", 4),
        ("curvature_calls", 4),
        ("conditioning_calls", 2),
        ("gradient_calls", True),
        ("accepted_proposals", 1.0),
    ],
)
def test_work_and_damping_tampering_is_rejected(field, value):
    after = report(1, 4, 1, 0.5, proposal())
    after[field] = value
    with pytest.raises(AssertionError):
        audit.verify_transition(report(0, 0, 0, 1.0, None), after)


def test_zero_gradient_exhausts_ladder_and_damping_clips():
    p = proposal((10.0,) * 5, predicted=0.0, selected_alpha=0.0)
    before = report(1, 6, 0, 1e8, p)
    after = report(2, 12, 0, 1e8, p)
    assert not audit.verify_transition(before, after)["accepted"]
    with pytest.raises(AssertionError):
        audit.accounting(report(0, 0, 0, 1.0, p), 0)


def endpoint_fixture():
    channels = 11
    terms = channels * (channels + 1) // 2
    values = dict(
        metadata=np.asarray(
            json.dumps(
                dict(
                    format="glassbox-online-fit-v8",
                    model=dict(dt_s=0.05, delay_steps=1),
                )
            )
        ),
        norm_body_mean=np.zeros(9),
        norm_body_scale=np.ones(9),
        norm_motion_bound_scale=np.full(6, 8.0),
        norm_input_mean=np.zeros(1),
        norm_input_scale=np.ones(1),
        norm_output_scale=np.ones(6),
        norm_quadratic_scale=np.ones(terms),
        param_quadratic=np.zeros((terms, 6)),
        scale=np.ones((1, 15)),
    )
    left, right = np.triu_indices(channels)
    values["param_quadratic"][(left == 0) & (right == 0), 0] = 1.0
    values["param_quadratic"][(left == 0) & (right == 9), 0] = 2.0
    predictions = {}
    for role, count, velocity, command in [
        ("bootstrap", 2, 2.0, 3.0),
        ("recent", 1, 1.0, 4.0),
    ]:
        states = np.zeros((count, 3, 15))
        states[..., 6:] = np.eye(3).ravel()
        states[..., 0] = velocity
        up = np.full((count, 2, 1), command)
        up[:, 0] = 99.0  # Pre-delay input is excluded from the v6 RMS domain.
        future = np.full((count, 1, 1), command)
        target = states[:, -1:].copy()
        for key, value in zip(
            ("past_states", "past_inputs", "future_inputs", "future_states"),
            (states, up, future, target),
        ):
            values[role + "_" + key] = value
        predictions[role] = target.copy()
    return values, predictions


def test_endpoint_uses_equal_roles_and_v6_rms_not_v7_envelope():
    values, predictions = endpoint_fixture()
    # Recent velocity residual has group radius 2, Huber 1.5, /3 groups /2 roles.
    predictions["recent"][..., 0] += 2.0
    result = audit.endpoint_components(values, predictions)
    motion, command = np.sqrt(2.5), np.sqrt(12.5)
    np.testing.assert_allclose(result["domain"][0], motion)
    np.testing.assert_allclose(np.array(result["domain"])[-2:], command)
    expected_prior = (
        0.01
        / 4
        * ((2 * motion**2 * 0.05) ** 2 + 2 * (2 * motion * command * 0.05) ** 2)
    )
    assert result["data_loss"] == 0.25
    np.testing.assert_allclose(result["prior_loss"], expected_prior)
    np.testing.assert_allclose(result["combined_loss"], 0.25 + expected_prior)
    transformed = copy.deepcopy(values)
    transformed["param_quadratic"] *= 3.0 / 2.0
    transformed["norm_quadratic_scale"] *= 3.0
    transformed["norm_output_scale"] *= 2.0
    np.testing.assert_allclose(
        audit.endpoint_components(transformed, predictions)["prior_loss"],
        expected_prior,
    )


def test_manifest_inventory_and_payload_authentication(tmp_path):
    (tmp_path / "data").write_bytes(b"fixture")
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(files={"data": audit.sha(tmp_path / "data")}))
    )
    authority = audit.sha(tmp_path / "manifest.json")
    assert audit.authenticate(tmp_path, authority) == 1
    (tmp_path / "data").write_bytes(b"altered")
    with pytest.raises(AssertionError, match="payload differs"):
        audit.authenticate(tmp_path, authority)
    (tmp_path / "extra").write_bytes(b"extra")
    with pytest.raises(AssertionError, match="inventory differs"):
        audit.authenticate(tmp_path, authority)


def test_verifier_source_has_no_model_or_jax_import():
    import ast

    tree = ast.parse(Path(audit.__file__).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module)
    assert not any(
        name.startswith(("glassbox", "jax", "evaluate_online")) for name in names
    )


def test_original_single_role_prior_fixture():
    assert audit.self_test()["analytic_fixture_passed"]


def test_journal_binds_order_scalar_decision_and_derived_prediction_arrays(tmp_path):
    norms = dict(
        feature_scale=np.ones(10), quadratic_scale=np.ones(3), output_scale=np.ones(6)
    )
    np.savez(tmp_path / "normalization.npz", **norms)
    np.savez(
        tmp_path / "final-online.npz",
        **{"norm_" + key: value for key, value in norms.items()},
    )
    before = dict(report(0, 0, 0, 1.0, None), cursor=7)
    after = dict(report(1, 4, 1, 0.5, proposal()), cursor=8)
    predictions = dict(
        origin=np.array([7]),
        truth=np.ones((1, 15)),
        candidate=np.zeros((1, 15), dtype=np.float32),
        frozen=np.zeros((1, 15), dtype=np.float32),
        kinematic=np.zeros((1, 15)),
        model_before=np.array(["before"]),
        model_after=np.array(["after"]),
        trial_evaluations=np.array([3]),
        selected_alpha=np.array([0.25]),
    )
    for key in audit.COUNTERS:
        predictions[key + "_before"] = np.array([before[key]])
        predictions[key + "_after"] = np.array([after[key]])
    source = dict(commands=np.ones((8, 1)))
    entries = [
        (
            dict(phase="predicted", index=7, model="before", report=before),
            dict(
                candidate=predictions["candidate"][0],
                frozen=predictions["frozen"][0],
                kinematic=predictions["kinematic"][0],
                command=source["commands"][7],
            ),
        ),
        (dict(phase="revealed", index=7), dict(truth=predictions["truth"][0])),
        (
            dict(phase="assimilated", index=7, model="after", report=after),
            {"norm_" + key: value for key, value in norms.items()},
        ),
    ]
    with (
        (tmp_path / "arrays.bin").open("wb") as arrays,
        (tmp_path / "events.jsonl").open("w") as journal,
    ):
        for event, values in entries:
            event["arrays"] = {}
            for key, value in values.items():
                event["arrays"][key] = arrays.tell()
                np.save(arrays, value, allow_pickle=False)
            journal.write(json.dumps(event) + "\n")
    info = dict(online_final_report=after)
    before_norms, before_reports, events, work = audit.journal(
        tmp_path, predictions, source, info, [7]
    )
    assert events == 3 and before_reports[7] == before
    np.testing.assert_array_equal(
        before_norms[7]["feature_scale"], norms["feature_scale"]
    )
    assert (
        work["total_objective_calls"] == 4
        and work["selected_alpha_histogram"]["0.25"] == 1
    )
    predictions["trial_evaluations"][0] = 1
    with pytest.raises(AssertionError, match="derived trial count"):
        audit.journal(tmp_path, predictions, source, info, [7])
    predictions["trial_evaluations"][0] = 3
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    (tmp_path / "events.jsonl").write_text(
        "\n".join([lines[1], lines[0], lines[2]]) + "\n"
    )
    with pytest.raises(AssertionError):
        audit.journal(tmp_path, predictions, source, info, [7])
