"""Synthetic bookkeeping fixtures only: no learner fit or simulator construction."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import collect_throw as collect
import evaluate_online as evaluate


def observations(n=3):
    result = np.zeros((n, 15))
    result[:, 6:] = np.eye(3).ravel()
    return result


def test_vector_rmse_is_not_component_mean():
    truth = observations()
    prediction = truth.copy()
    prediction[:, :3] = [3, 4, 0]
    prediction[:, 3:6] = [0, 0, 2]
    actual = evaluate.metrics(prediction, truth)
    assert actual["velocity_rmse_m_s"] == 5
    assert actual["body_rate_rmse_rad_s"] == 2
    assert actual["orientation_rmse_rad"] == 0


def test_geodesic_keeps_small_angles_without_float32_trace_floor():
    truth = observations(1)
    prediction = truth.astype(np.float32)
    prediction[0, 6:] = Rotation.from_rotvec([0, 0, 1e-7]).as_matrix().ravel()
    actual = evaluate.metrics(prediction, truth)["orientation_rmse_rad"]
    assert actual == pytest.approx(1e-7, rel=1e-6)
    prediction[0, 6:] = (np.eye(3) * (1 + 1e-6)).ravel()
    assert evaluate.metrics(prediction, truth)["orientation_rmse_rad"] < 1e-12


def test_kinematics_integrates_body_rate_but_holds_velocity_and_rate():
    x = observations(1)[0]
    x[:6] = [2, 3, 4, 0, 0, 2]
    prediction = evaluate.kinematic(x, 0.1)
    np.testing.assert_array_equal(prediction[:6], x[:6])
    assert Rotation.from_matrix(
        prediction[6:].reshape(3, 3)
    ).magnitude() == pytest.approx(0.2)


def aggregate_case(name, family, value, complete=True):
    return dict(
        id=name,
        family=family,
        complete=complete,
        ratios={"velocity_rmse_m_s": value, "body_rate_rmse_rad_s": value},
        latency={"observe": {"warm_p95_s": 0.001}},
        prefix={"dt_s": 0.01},
    )


def test_aggregation_equal_family_weights_and_no_single_cell_veto():
    cases = [
        aggregate_case(name, "quad", 0.25)
        for name in ("quad-arm-115", "quad-arm-125", "quad-arm-135", "quad-change")
    ]
    cases += [aggregate_case(f"fixedwing-{i}", "fixedwing", 0.81) for i in (80, 81)]
    result = evaluate.aggregate(cases)
    assert result["aggregate_ratio"] == pytest.approx(0.45)
    assert result["accuracy_passed"]
    cases[0]["ratios"] = {"velocity_rmse_m_s": 2.0, "body_rate_rmse_rad_s": 2.0}
    assert evaluate.aggregate(cases)["accuracy_passed"]
    cases[0]["complete"] = False
    assert not evaluate.aggregate(cases)["accuracy_passed"]
    cases[0]["complete"] = True
    cases[0]["id"] = cases[1]["id"]
    assert not evaluate.aggregate(cases)["accuracy_passed"]


class FakeOnline:
    """Deterministic API stand-in; holds motion and counts assimilation only."""

    def __init__(self, prefix):
        segment = prefix.segments[0]
        self.cursor = segment.start_row + len(segment.inputs)
        self.initial_cursor = self.cursor
        self.count = 0
        self.history = round(0.5 / segment.dt_s)
        self.norms = dict(
            body_mean=np.zeros(9),
            body_scale=np.ones(9),
            motion_bound_scale=np.full(6, 4.0),
            input_mean=np.zeros(segment.inputs.shape[1]),
            input_scale=np.ones(segment.inputs.shape[1]),
        )
        self.predicted = False

    @property
    def model(self):
        return SimpleNamespace(
            params={"w": np.asarray([float(self.count)])},
            norms=copy.deepcopy(self.norms),
            history_steps=self.history,
            fingerprint=hashlib.sha256(str(self.count).encode()).hexdigest(),
        )

    @property
    def report(self):
        return dict(
            cursor=self.cursor,
            observations=self.count,
            optimizer_steps=4 * self.count,
            gradient_calls=4 * self.count,
            objective_calls=8 * self.count,
            accepted_proposals=4 * self.count,
        )

    def predict(self, past, inputs, future):
        assert len(past) == len(inputs) + 1 and len(future) == 1
        self.predicted = True
        return past[-1:].astype(np.float32)

    def observe(self, index, command, observation):
        assert self.predicted and index == self.cursor and observation.shape == (15,)
        self.cursor += 1
        self.count += 1
        self.predicted = False

    def save(self, path):
        np.savez_compressed(
            path,
            param_w=self.model.params["w"],
            **{"norm_" + key: value for key, value in self.norms.items()},
        )


def run_fixture(tmp_path, monkeypatch, *, fail=False, short=False):
    import glassbox

    class Chosen(FakeOnline):
        def observe(self, *args):
            if fail:
                raise RuntimeError("synthetic update failure")
            super().observe(*args)

    monkeypatch.setattr(glassbox, "OnlineFit", Chosen)
    count = 10 if short else 20
    native = np.zeros((count + 1, 13))
    native[:, 6] = 1
    native[:, 3] = np.arange(count + 1) * 0.05
    source = tmp_path / "stream.npz"
    np.savez_compressed(
        source,
        states=native,
        commands=np.zeros((count, 3)),
        time_s=np.arange(count + 1) * 0.05,
    )
    info = dict(
        id="fixture",
        family="fixedwing",
        opaque_id="opaque",
        dt_s=0.05,
        begin=0,
        first=15,
        ordered_commands=["throttle [1]", "aileron [rad]", "elevator [rad]"],
    )
    output = tmp_path / "case"
    result = evaluate.evaluate_case(source, output, info)
    return result, output, evaluate.arrays(source)


def test_causal_journal_and_exact_accounting_without_fitting(tmp_path, monkeypatch):
    result, output, source = run_fixture(tmp_path, monkeypatch)
    assert result["complete"] and result["predicted"] == result["assimilated"] == 5
    data, info = (
        evaluate.arrays(output / "predictions.npz"),
        evaluate.read(output / "case.json"),
    )
    evaluate.verify_journal(output, data, source, info)
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    assert [row["phase"] for row in events] == [
        "predicted",
        "revealed",
        "assimilated",
    ] * 5
    events[0]["phase"] = "revealed"
    (output / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events))
    with pytest.raises(ValueError, match="journal order"):
        evaluate.verify_journal(output, data, source, info)


@pytest.mark.parametrize("short", [False, True])
def test_partial_failure_preserves_missing_rows_and_checkpoint(
    tmp_path, monkeypatch, short
):
    result, output, source = run_fixture(tmp_path, monkeypatch, fail=True, short=short)
    assert not result["complete"]
    assert result["assimilated"] == 0
    assert (output / "predictions.npz").is_file()
    data, info = (
        evaluate.arrays(output / "predictions.npz"),
        evaluate.read(output / "case.json"),
    )
    assert info["status"] == "failed"
    evaluate.verify_journal(output, data, source, info)


def test_sealed_numeric_mutation_is_rejected_even_with_new_local_checksum(tmp_path):
    np.savez(tmp_path / "values.npz", x=np.zeros(3))
    authority = collect.seal(tmp_path, "fixture")
    collect.authenticate(tmp_path, authority)
    np.savez(tmp_path / "values.npz", x=np.ones(3))
    with pytest.raises(ValueError, match="artifact hash"):
        collect.authenticate(tmp_path, authority)
    manifest = collect.read(tmp_path / "manifest.json")
    manifest["files"]["values.npz"] = collect.digest(tmp_path / "values.npz")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest authority"):
        collect.authenticate(tmp_path, authority)


def test_output_is_exclusive(tmp_path):
    with pytest.raises(FileExistsError):
        collect.run(tmp_path)
    with pytest.raises(FileExistsError):
        evaluate.run(tmp_path, "unused", tmp_path)
    assert not list(tmp_path.iterdir())
