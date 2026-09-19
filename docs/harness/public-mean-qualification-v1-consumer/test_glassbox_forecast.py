"""Consumer-boundary tests; no learned fit, optimizer or simulator is used."""

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from crazydart import glassbox_forecast as consumer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(path, data):
    path.write_text(json.dumps(data, sort_keys=True, allow_nan=False))
    return sha(path)


class PublicModel:
    """A small public-interface test double, not a fitted Glassbox artifact."""

    def __init__(self, identity):
        self.identity = identity
        self.contract = dict(
            configuration_id="boundary-fixture",
            state_channels=["x [m,fixture]", "v [m/s,fixture]"],
            input_channels=["u [unit,fixture]"],
            dt_s=0.1,
        )
        self.recipe = {"id": "generic-memory-v4-prototype"}
        self.history_steps = self.horizon_steps = 2
        self.predict_calls = 0

    def fingerprint(self):
        return hashlib.sha256(self.identity.encode()).hexdigest()

    def predict(self, past_states, past_inputs, future_inputs):
        self.predict_calls += 1
        x, u = jnp.asarray(past_states), jnp.asarray(future_inputs)
        return x[-1] + jnp.cumsum(u, axis=0) * jnp.array([1.0, 0.5])

    def envelope(self, horizon_steps=None):
        return np.full((horizon_steps or 2, 2), 0.125)


@pytest.fixture
def packet(tmp_path, monkeypatch):
    root = tmp_path / "packet"
    root.mkdir()
    data = dict(
        format=consumer.FORMAT,
        qualification=dict(
            protocol_sha256=consumer.PROTOCOL_SHA256,
            implementation_commit="1" * 40,
            exporter_module=consumer.EXPORTER_MODULE,
            exporter_source_sha256="2" * 64,
            query_selection_spec_sha256=consumer.SELECTION_SHA256,
            test_seed_shift_from_original=11000000,
        ),
        models=[],
        recordings=[],
        queries=[],
    )
    model_objects, collections, loads = {}, {}, Counter()
    for identity in consumer.MODEL_IDS:
        model = PublicModel(identity)
        model_objects[identity] = model
        name = identity + "-model.npz"
        np.savez(root / name, test_double=np.zeros(1))
        data["models"].append(
            dict(
                id=identity,
                path=name,
                sha256=sha(root / name),
                fingerprint=model.fingerprint(),
                contract=model.contract.copy(),
                history_steps=2,
                horizon_steps=2,
            )
        )
        parents, segments = [], []
        for index, (scope, kind) in enumerate(
            (scope, kind)
            for scope in ("primary", "heading_shift")
            for kind in ("factual", "response")
            for _ in range(2)
        ):
            parent = f"parent-{index:02d}"
            x = np.column_stack((np.arange(10.0), np.arange(10.0) + 1))
            u = np.full((9, 1), 0.25)
            segment = SimpleNamespace(
                recording_id=parent,
                segment_id="whole",
                start_row=0,
                states=x,
                inputs=u,
                dt_s=0.1,
            )
            segments.append(segment)
            parents.append(
                dict(
                    recording_id=parent,
                    source_parent_sha256="3" * 64,
                    source_record_entry_sha256="4" * 64,
                    segments=[
                        dict(
                            segment_id="whole", start_row=0, state_rows=10, input_rows=9
                        )
                    ],
                )
            )
            arrays = dict(
                past_states=x[2:5].copy(),
                past_inputs=u[2:4].copy(),
                future_inputs=u[4:6].copy(),
            )
            if kind == "response":
                arrays["factual_inputs"] = arrays["future_inputs"].copy()
                arrays["future_inputs"][0, 0] += 0.125
            name = f"{identity}-query-{index:02d}.npz"
            np.savez(root / name, **arrays)
            data["queries"].append(
                dict(
                    model_id=identity,
                    recordings_id=identity + "-recordings",
                    parent=parent,
                    id=f"query-{index:02d}",
                    segment_id="whole",
                    source_origin=4,
                    kind=kind,
                    scope=scope,
                    dt_s=0.1,
                    horizon_steps=2,
                    branches=["factual"]
                    if kind == "factual"
                    else ["intervened", "factual"],
                    path=name,
                    sha256=sha(root / name),
                    source_query_sha256="5" * 64,
                )
            )
        name = identity + "-recordings.npz"
        np.savez(root / name, test_double=np.zeros(1))
        data["recordings"].append(
            dict(
                id=identity + "-recordings",
                model_id=identity,
                path=name,
                sha256=sha(root / name),
                parents=parents,
            )
        )
        collections[name] = SimpleNamespace(
            segments=tuple(segments),
            configuration_id=model.contract["configuration_id"],
            state_channels=tuple(model.contract["state_channels"]),
            input_channels=tuple(model.contract["input_channels"]),
        )
    data["queries"].sort(key=lambda row: (row["model_id"], row["parent"], row["id"]))
    manifest = root / "INPUT.json"

    def load_model(path):
        identity = Path(path).name.removesuffix("-model.npz")
        loads[identity] += 1
        return model_objects[identity]

    monkeypatch.setattr(consumer, "LearnedDynamics", SimpleNamespace(load=load_model))
    monkeypatch.setattr(
        consumer, "load_recordings", lambda path: collections[Path(path).name]
    )
    return SimpleNamespace(
        root=root,
        manifest=manifest,
        data=data,
        models=model_objects,
        collections=collections,
        loads=loads,
        expected=seal(manifest, data),
    )


def mutate_query(packet, change):
    row = packet.data["queries"][0]
    path = packet.root / row["path"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    change(arrays)
    np.savez(path, **arrays)
    row["sha256"] = sha(path)
    return seal(packet.manifest, packet.data)


def assert_no_prediction(packet):
    assert all(model.predict_calls == 0 for model in packet.models.values())


def test_complete_packet_loads_each_model_once_and_reconstructs_all_histories(packet):
    value = consumer.validate_packet(packet.manifest, packet.expected)
    assert len(value.queries) == 16
    assert packet.loads == {"crazyflow": 1, "cascade": 1}
    assert all(checked == 2 for _, _, checked in value.queries)
    assert_no_prediction(packet)


@pytest.mark.parametrize(
    "mutation",
    ("fingerprint", "channels", "dt", "origin", "future_raw", "future_coherent"),
)
def test_required_negative_cases_fail_before_forecast(packet, mutation):
    if mutation == "fingerprint":
        packet.data["models"][0]["fingerprint"] = "f" * 64
    elif mutation == "channels":
        packet.data["models"][0]["contract"]["state_channels"] = list(
            reversed(packet.data["models"][0]["contract"]["state_channels"])
        )
    elif mutation == "dt":
        packet.data["queries"][0]["dt_s"] = 0.2
    elif mutation == "origin":
        packet.data["queries"][0]["source_origin"] = 1
    else:
        row = packet.data["queries"][0]
        previous = row["sha256"]
        mutate_query(
            packet, lambda arrays: arrays["future_inputs"].__setitem__((0, 0), 9.0)
        )
        if mutation == "future_raw":
            row["sha256"] = previous
    expected = seal(packet.manifest, packet.data)
    with pytest.raises(consumer.PacketError):
        consumer.validate_packet(packet.manifest, expected)
    assert_no_prediction(packet)


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_key",
        "boolean_origin",
        "unknown_scope",
        "missing_stratum",
        "extra_file",
        "duplicate_path",
        "escaping_path",
        "symlink",
        "wrong_root_sha",
    ),
)
def test_schema_roster_and_path_rejections_precede_forecasts(packet, mutation):
    if mutation == "extra_key":
        packet.data["expected_means"] = []
    elif mutation == "boolean_origin":
        packet.data["queries"][0]["source_origin"] = True
    elif mutation == "unknown_scope":
        packet.data["queries"][0]["scope"] = "other"
    elif mutation == "missing_stratum":
        packet.data["queries"][0]["scope"] = "wind_shift"
    elif mutation == "extra_file":
        (packet.root / "expected-output.txt").write_text("forbidden")
    elif mutation == "duplicate_path":
        packet.data["queries"][0]["path"] = packet.data["queries"][1]["path"]
    elif mutation == "escaping_path":
        packet.data["queries"][0]["path"] = "../outside.npz"
    elif mutation == "symlink":
        original = packet.root / packet.data["queries"][0]["path"]
        target = packet.root.parent / "outside.npz"
        original.rename(target)
        original.symlink_to(target)
    expected = seal(packet.manifest, packet.data)
    if mutation == "wrong_root_sha":
        expected = "0" * 64
    with pytest.raises(consumer.PacketError):
        consumer.validate_packet(packet.manifest, expected)
    assert_no_prediction(packet)


@pytest.mark.parametrize(
    "mutation",
    ("target", "float32", "nonfinite", "fortran", "history", "duplicate_zip"),
)
def test_input_only_array_contract_is_strict(packet, mutation):
    if mutation == "duplicate_zip":
        row = packet.data["queries"][0]
        path = packet.root / row["path"]
        with zipfile.ZipFile(path) as archive:
            raw = archive.read("future_inputs.npy")
        with pytest.warns(UserWarning), zipfile.ZipFile(path, "a") as archive:
            archive.writestr("future_inputs.npy", raw)
        row["sha256"] = sha(path)
        expected = seal(packet.manifest, packet.data)
    else:

        def change(arrays):
            if mutation == "target":
                arrays["target"] = np.zeros((2, 2))
            elif mutation == "float32":
                arrays["future_inputs"] = arrays["future_inputs"].astype(np.float32)
            elif mutation == "nonfinite":
                arrays["future_inputs"][0, 0] = np.nan
            elif mutation == "fortran":
                arrays["past_states"] = np.asfortranarray(arrays["past_states"])
            elif mutation == "history":
                arrays["past_states"][0, 0] += 1

        expected = mutate_query(packet, change)
    with pytest.raises(consumer.PacketError):
        consumer.validate_packet(packet.manifest, expected)
    assert_no_prediction(packet)


@pytest.mark.parametrize("suffix", ('"format":"duplicate"', '"unused":NaN'))
def test_json_duplicate_and_nonfinite_constants_rejected(packet, suffix):
    raw = packet.manifest.read_text()
    packet.manifest.write_text(raw[:-1] + "," + suffix + "}")
    with pytest.raises(consumer.PacketError):
        consumer.validate_packet(packet.manifest, sha(packet.manifest))
    assert packet.loads == {}


def test_recording_segment_manifest_is_not_a_provenance_substitute(packet):
    packet.collections["cascade-recordings.npz"].segments[0].start_row = 1
    with pytest.raises(consumer.PacketError, match="segment roster"):
        consumer.validate_packet(packet.manifest, packet.expected)
    assert_no_prediction(packet)


def test_all_public_outputs_and_response_pairs_use_the_fixed_roster(
    packet, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        consumer, "observed_runtime", lambda: {"test_double_runtime": True}
    )
    output = tmp_path / "output"
    with jax.enable_x64(False):
        result = consumer.run(packet.manifest, packet.expected, output)
    assert result["counts"] == dict(
        models=2,
        queries=16,
        means=24,
        responses=8,
        fits=0,
        updates=0,
        initializers=0,
        optimizer_steps=0,
        simulator_calls=0,
    )
    assert packet.loads == {"crazyflow": 1, "cascade": 1}
    assert result["arrays_sha256"] == sha(output / "arrays.npz")
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        for row in result["means"]:
            assert row["dtypes"]["eager"] == "<f4"
            np.testing.assert_array_equal(
                arrays[row["arrays"]["jvp"]], [[1.0, 0.5], [1.0, 0.5]]
            )
            np.testing.assert_array_equal(
                arrays[row["arrays"]["tangent"]], [[1.0], [0.0]]
            )
        for row in result["responses"]:
            a, b = (
                result["means"][row[key]] for key in ("intervened_mean", "factual_mean")
            )
            for name in ("eager", "compiled"):
                np.testing.assert_array_equal(
                    arrays[row["arrays"][name]],
                    arrays[a["arrays"][name]] - arrays[b["arrays"][name]],
                )
    assert sha(packet.manifest) == packet.expected


def test_bad_final_query_cannot_leave_partial_predictions(
    packet, tmp_path, monkeypatch
):
    monkeypatch.setattr(consumer, "observed_runtime", lambda: {})
    packet.data["queries"][-1]["source_origin"] = 1
    expected = seal(packet.manifest, packet.data)
    with pytest.raises(consumer.PacketError):
        consumer.run(packet.manifest, expected, tmp_path / "output")
    assert_no_prediction(packet)
    assert not (tmp_path / "output").exists()


def test_output_inside_packet_rejected_before_forecast(packet, monkeypatch):
    monkeypatch.setattr(consumer, "observed_runtime", lambda: {})
    with pytest.raises(consumer.PacketError, match="separate"):
        consumer.run(packet.manifest, packet.expected, packet.root / "output")
    assert_no_prediction(packet)
