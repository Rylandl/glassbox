"""No-fit task qualification, exercised without installing the flight simulator."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.control.plan import Prediction, ReferenceTrajectory
from glassbox.experimental import task_qualification as q
from glassbox.experimental.harness import control_pass_criterion, control_reference
from glassbox.experimental.learned_plan import observed_from_state
from glassbox.experimental.qualification_control import OracleArm
from glassbox.learner import RECIPE

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "docs/harness/controller-task-v1.json").read_text())
CONTROL = json.loads((ROOT / "docs/harness/control-v5.json").read_text())
INITIAL = np.array([0, 0, 100, 18, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
COMMAND = np.array([0.5, 0.0, 0.0])


class Equations:
    def __init__(self):
        self.compile_signature = "task-qualification-test-equations-v1"

        def advance(state, command):
            velocity = state[3:6] + 0.05 * command
            position = state[:3] + 0.025 * (state[3:6] + velocity)
            return state.at[:3].set(position).at[3:6].set(velocity)

        def predict(state, commands):
            def step(current, command):
                future = advance(current, command)
                return future, observed_from_state(future)

            return jax.lax.scan(step, state, commands)[1]

        self.advance = jax.jit(advance)
        self.predict = jax.jit(predict)
        self.reset = lambda state, command: jnp.asarray(state)
        self.canonical = lambda state: state


@pytest.fixture
def learned():
    model = SimpleNamespace(
        params={"memory": np.zeros((4, 8))},
        norms={},
        delay_steps=2,
        metadata=lambda: {"fixture": "task-qualification"},
        arrays=lambda: {"memory": np.zeros((4, 8))},
        memory_state=lambda states, commands: jnp.zeros(8),
    )
    return SimpleNamespace(
        _model=model,
        contract={
            "dt_s": 0.05,
            "state_channels": CONTROL["telemetry"]["observed_channels"],
            "input_channels": ["throttle", "roll", "pitch"],
        },
        recipe=dict(RECIPE),
        history_steps=10,
        horizon_steps=5,
        envelope=lambda steps: np.full((steps, 15), 0.2),
    )


def make_arm(learned, name=q.BASELINE, equations=None):
    return q.TaskOracleArm(
        PLAN,
        CONTROL,
        learned,
        None,
        name,
        _equations=Equations() if equations is None else equations,
    )


def short_trial(arm):
    arm.reset(INITIAL, COMMAND)
    state, previous, warm = INITIAL, COMMAND, None
    states, commands = [state], []
    for index in range(4):
        arm.observe(state)
        if arm.ready:
            reference = ReferenceTrajectory(
                control_reference(
                    INITIAL,
                    (index + np.arange(6)) * 0.05,
                    CONTROL["tracking_reference"],
                )
            )
            result = arm.solve(state, reference, previous, warm_start=warm)
            command, warm = np.asarray(result.command), result.warm_start
        else:
            command = previous
        state = np.asarray(
            arm.equations.advance(jnp.asarray(state), jnp.asarray(command))
        )
        arm.command_applied(command)
        states.append(state)
        commands.append(command)
        previous = command
    return dict(
        states=np.asarray(states),
        commands=np.asarray(commands),
        initial_command=COMMAND,
        reference_anchor_state=INITIAL,
    )


def test_plan_pins_every_inherited_source_and_eight_trials():
    plan, raw = q.frozen_plan()
    assert q.digest(raw) == q.PLAN_SHA256
    assert plan["no_fit"] is True
    assert plan["control_qualification"]["seeds"] == [101, 102, 104, 105]
    names = q.artifact_names(plan)
    assert len([name for name in names if name.endswith("oracle.npz")]) == 8
    assert len([name for name in names if name.endswith("tracking.npz")]) == 8


def test_runner_contains_no_fitting_or_update_call():
    tree = ast.parse(Path(q.__file__).read_text())
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        (isinstance(f, ast.Name) and f.id in {"fit", "update"})
        or (isinstance(f, ast.Attribute) and f.attr in {"fit", "update_recordings"})
        for f in calls
    )


def test_candidate_changes_only_position_scales_and_signatures(learned):
    equations = Equations()
    original = OracleArm(CONTROL, learned, None, _equations=equations)
    baseline = make_arm(learned, equations=equations)
    candidate = make_arm(learned, q.CANDIDATE, equations)
    assert baseline.plan.compile_signature == original.plan.compile_signature
    assert baseline.plan.base.compile_signature == original.plan.base.compile_signature
    assert candidate.plan.compile_signature != baseline.plan.compile_signature
    assert candidate.plan.base.compile_signature != baseline.plan.base.compile_signature
    assert baseline.plan.tolerances.position_m == (5.0, 5.0, 3.0)
    assert candidate.plan.tolerances.position_m == (5.0, 0.5, 0.5)
    for key in ("velocity_m_s", "attitude_rad", "angular_velocity_rad_s"):
        assert getattr(candidate.plan.tolerances, key) == getattr(
            baseline.plan.tolerances, key
        )
    assert candidate.plan.policy == baseline.plan.policy == original.plan.policy
    assert candidate.plan.equations is baseline.plan.equations
    assert candidate.plan.base.learned is baseline.plan.base.learned
    assert candidate.plan.safety_envelope == baseline.plan.safety_envelope
    for key in ("command_minimum", "command_maximum"):
        np.testing.assert_array_equal(
            getattr(candidate.plan, key), getattr(baseline.plan, key)
        )
    np.testing.assert_array_equal(
        candidate.plan.values.forecast_error_covariance,
        baseline.plan.values.forecast_error_covariance,
    )
    assert candidate.constant_objective > baseline.constant_objective


def test_baseline_commands_and_original_diagnostics_are_exact(learned):
    equations = Equations()
    old = OracleArm(CONTROL, learned, None, _equations=equations)
    new = make_arm(learned, equations=equations)
    a, b = short_trial(old), short_trial(new)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    saved = new.diagnostic_arrays()
    for key, value in old.diagnostic_arrays().items():
        np.testing.assert_array_equal(saved[key], value)


@pytest.mark.parametrize("name", [q.BASELINE, q.CANDIDATE])
def test_replay_recomputes_every_solver_diagnostic(learned, name):
    arm = make_arm(learned, name)
    tracking = short_trial(arm)
    diagnostics = arm.diagnostic_arrays()
    assert set(diagnostics) == q.ORIGINAL_DIAGNOSTICS | q.EXTRA_DIAGNOSTICS
    checked = q.verify_solves(PLAN, arm, CONTROL, tracking, diagnostics)
    assert checked["verified_optimizer_solves"] == 2
    assert checked["verified_oracle_forecasts"] == 2
    assert checked["solver_statuses"]["model_not_ready"] == 2
    np.testing.assert_array_equal(
        diagnostics["variable_final_objectives"],
        diagnostics["final_objectives"] - diagnostics["constant_covariance_objectives"],
    )


@pytest.mark.parametrize("field", sorted(q.ORIGINAL_DIAGNOSTICS | q.EXTRA_DIAGNOSTICS))
def test_replay_rejects_altered_diagnostic_even_without_outer_hash(learned, field):
    arm = make_arm(learned, q.CANDIDATE)
    tracking = short_trial(arm)
    diagnostics = arm.diagnostic_arrays()
    changed = diagnostics[field].copy()
    if changed.dtype.kind == "U":
        changed.flat[0] = "forged"
    elif changed.dtype.kind == "b":
        changed.flat[0] = not changed.flat[0]
    else:
        changed.flat[0] += 1 if changed.dtype.kind in "iu" else 0.1
    diagnostics[field] = changed
    with pytest.raises((ValueError, AssertionError)):
        q.verify_solves(PLAN, arm, CONTROL, tracking, diagnostics)


def test_replay_rejects_changed_issued_command(learned):
    arm = make_arm(learned)
    tracking = short_trial(arm)
    diagnostics = arm.diagnostic_arrays()
    tracking["commands"][2, 0] += 0.01
    with pytest.raises((ValueError, AssertionError)):
        q.verify_solves(PLAN, arm, CONTROL, tracking, diagnostics)


def test_replay_rejects_missing_diagnostic_and_wrong_shapes(learned):
    arm = make_arm(learned)
    tracking = short_trial(arm)
    diagnostics = arm.diagnostic_arrays()
    del diagnostics["iterations"]
    with pytest.raises(ValueError, match="fields"):
        q.verify_solves(PLAN, arm, CONTROL, tracking, diagnostics)
    diagnostics = arm.diagnostic_arrays()
    diagnostics["statuses"] = diagnostics["statuses"][:1]
    with pytest.raises(ValueError, match="shape"):
        q.verify_solves(PLAN, arm, CONTROL, tracking, diagnostics)


def test_input_snapshot_preflights_hashes_and_keeps_original_bytes(tmp_path):
    (tmp_path / "model.npz").write_bytes(b"original")
    plan = {
        "input_paths": {"control/model.npz": "model.npz"},
        "input_sha256": {"control/model.npz": q.digest(b"original")},
    }
    inputs = q.input_snapshot(plan, tmp_path)
    (tmp_path / "model.npz").write_bytes(b"altered")
    assert inputs["control/model.npz"] == b"original"
    with pytest.raises(ValueError, match="input changed"):
        q.input_snapshot(plan, tmp_path)


def rows_for_report():
    rows = []
    for repetition, order in enumerate(PLAN["control_qualification"]["arm_order"]):
        for name in order:
            row = dict.fromkeys(q.ROW_FIELDS)
            row |= dict(
                arm=name,
                repetition=repetition,
                initial_state_seed=PLAN["control_qualification"]["seeds"][repetition],
                directory=f"trial-{repetition}/{name}",
                terminated=False,
                fallback_count=0,
                maximum_command_bound_violation=0.0,
                pass_criterion=dict(
                    met=name == q.CANDIDATE,
                    within_tolerance_fraction=1.0 if name == q.CANDIDATE else 0.4,
                    lateral_rmse_m=0.3,
                    altitude_rmse_m=0.2,
                ),
            )
            rows.append(row)
    return rows


def test_report_requires_each_candidate_to_meet_task_without_fallback():
    rows = rows_for_report()
    assert q.report(PLAN, rows, ["checked"])["candidate_qualifies"]
    for key, value in (
        ("terminated", True),
        ("fallback_count", 1),
        ("maximum_command_bound_violation", 0.1),
    ):
        altered = copy.deepcopy(rows)
        next(r for r in altered if r["arm"] == q.CANDIDATE)[key] = value
        assert not q.report(PLAN, altered, ["checked"])["candidate_qualifies"]
    next(r for r in reversed(rows) if r["arm"] == q.CANDIDATE)["pass_criterion"][
        "met"
    ] = False
    assert not q.report(PLAN, rows, ["checked"])["candidate_qualifies"]


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "seed", "extra_field", "directory"]
)
def test_report_rejects_incomplete_or_forged_roster(change):
    rows = rows_for_report()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "seed":
        rows[0]["initial_state_seed"] = 103
    elif change == "extra_field":
        rows[0]["unfrozen"] = True
    else:
        rows[0]["directory"] = "other"
    with pytest.raises(ValueError):
        q.report(PLAN, rows, [])


def test_task_boundary_and_missing_samples_preserve_original_requirement():
    control = copy.deepcopy(CONTROL)
    # Use an exactly representable boundary; adding 0.5 to arbitrary sine
    # samples can round their subtraction a last bit above the threshold.
    control["tracking_reference"]["lateral_amplitude_m"] = 0.0
    control["tracking_reference"]["altitude_amplitude_m"] = 0.0
    times = np.arange(321) * 0.05
    target = control_reference(INITIAL, times, control["tracking_reference"])
    states = target.copy()
    states[:, 1:3] += 0.5
    at_boundary = control_pass_criterion(states, INITIAL, control)
    assert at_boundary["met"]
    assert at_boundary["within_samples"] == 281
    states[:, 1] += 1e-6
    assert not control_pass_criterion(states, INITIAL, control)["met"]
    shortened = control_pass_criterion(target[:100], INITIAL, control)
    assert shortened["scored_samples"] == 281
    assert shortened["terminated"]
    assert not shortened["met"]


@pytest.mark.parametrize("coordinate,ratio", [(1, 100.0), (2, 36.0)])
def test_task_scales_change_only_declared_position_penalty(learned, coordinate, ratio):
    reference = jnp.tile(jnp.asarray(INITIAL), (6, 1))
    states = reference.at[1:, coordinate].add(0.5)
    prediction = Prediction(
        mean_states=states,
        tangent_covariance=jnp.zeros((5, 12, 12)),
        commands=jnp.tile(jnp.asarray(COMMAND), (5, 1)),
        latent_states=jnp.zeros((6, 0)),
        exogenous=jnp.zeros((5, 0)),
    )
    costs = []
    for name in (q.BASELINE, q.CANDIDATE):
        arm = make_arm(learned, name)
        costs.append(
            float(
                arm.plan.stage_cost(
                    prediction, reference, jnp.asarray(COMMAND), arm.policy
                )
            )
        )
    np.testing.assert_allclose(costs[1] / costs[0], ratio, rtol=1e-6)


def test_frozen_plan_rejects_inherited_source_change(tmp_path, monkeypatch):
    relative = next(iter(PLAN["source_sha256"]))
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("# altered numerical dependency\n")
    monkeypatch.setattr(q, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="inherited source changed"):
        q.frozen_plan()


def test_environment_check_rejects_unpinned_runtime():
    changed = copy.deepcopy(PLAN)
    changed["environment"]["python"] = "0.0.0"
    with pytest.raises(ValueError, match="environment"):
        q.check_environment(changed)


@pytest.mark.parametrize("completed", [0, 3])
def test_failed_trial_keeps_and_replays_finite_prefix(
    learned, tmp_path, monkeypatch, completed
):
    from glassbox.experimental import harness

    equations = Equations()
    seed = 104
    with jax.enable_x64(True):
        initial = harness.control_initial_state(CONTROL, INITIAL, seed)

    class FailingPlant:
        def __init__(self, state, command, fail_after):
            self.initial_state = np.asarray(state).copy()
            self.initial_command = np.asarray(command).copy()
            self.state = jnp.asarray(state)
            self.count = 0
            self.fail_after = fail_after

        def advance(self, command):
            if self.count == self.fail_after:
                return np.full(13, np.nan)
            self.count += 1
            self.state = equations.advance(self.state, jnp.asarray(command))
            return np.asarray(self.state)

    arm = make_arm(learned, q.CANDIDATE, equations)
    prefix = "trial-2/oracle_task_scales"
    directory = tmp_path / prefix
    row = harness._control_trial(
        CONTROL,
        arm,
        FailingPlant(initial, COMMAND, completed),
        lambda times: control_reference(INITIAL, times, CONTROL["tracking_reference"]),
        INITIAL,
        directory,
    )
    row |= dict(repetition=2, initial_state_seed=seed, directory=prefix)
    assert row["completed_intervals"] == completed
    assert row["terminated"]
    assert not row["pass_criterion"]["met"]
    assert row["pass_criterion"]["scored_samples"] == 281
    diagnostics = arm.diagnostic_arrays()
    saved = {
        prefix + "/" + name: (directory / name).read_bytes()
        for name in ("tracking.npz", "timing.npz")
    }
    context = SimpleNamespace(
        manifest=CONTROL, initial_state=INITIAL, initial_command=COMMAND
    )
    monkeypatch.setattr(
        harness,
        "_control_tracking_plant",
        lambda manifest, state, command: FailingPlant(state, command, completed),
    )
    physical = q.replay_failed_prefix(PLAN, context, row, saved, diagnostics)
    assert physical["state_difference"] == 0
    tracking = q.arrays(saved[prefix + "/tracking.npz"])
    tracking["initial_command"] = COMMAND
    replay = q.verify_solves(PLAN, arm, CONTROL, tracking, diagnostics)
    assert replay["verified_optimizer_solves"] == max(0, completed + 1 - 2)
    assert replay["solver_statuses"] == row["solver_statuses"]

    # A claimed early termination is not evidence if the same next interval
    # actually remains finite under replay.
    monkeypatch.setattr(
        harness,
        "_control_tracking_plant",
        lambda manifest, state, command: FailingPlant(state, command, None),
    )
    with pytest.raises(ValueError, match="termination does not reproduce"):
        q.replay_failed_prefix(PLAN, context, row, saved, diagnostics)
