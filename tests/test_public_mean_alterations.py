"""Bounded orchestration/repair contracts, not additional alteration cases."""

from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental import public_mean_alterations as subject


def test_exact_frozen_roster_and_process_precision():
    assert list(subject.CHECKS) == [f"PM{i:02d}" for i in range(1, 13)]
    assert subject.case_role("PM06") == subject.case_role("PM10") == "historical"
    assert subject.case_role("PM08") == subject.case_role("PM11") == "public32"
    assert subject.case_role("PM02") == subject.case_role("PM05") == "public64"


def test_only_declared_typed_semantic_failures_are_translated():
    class HelperError(ValueError):
        check_id = "truth_roster_and_masks"

    def helper():
        raise HelperError("actual reconstructed query differs")

    with pytest.raises(subject.CheckFailure) as caught:
        subject.translate(
            helper, source_id="truth_roster_and_masks", target_id="query_reconstruction"
        )
    assert caught.value.check_id == "query_reconstruction"
    with pytest.raises(HelperError):
        subject.translate(
            helper, source_id="another_check", target_id="query_reconstruction"
        )
    with pytest.raises(KeyError):
        subject.translate(
            lambda: {}["missing"],
            source_id="truth_roster_and_masks",
            target_id="query_reconstruction",
        )


def test_external_anchor_checks_payloads_even_when_root_file_unchanged(tmp_path):
    payload = tmp_path / "payload"
    payload.write_bytes(b"original")
    subject.write(
        tmp_path / "ROOT.json", {"files": {"payload": subject.digest(payload)}}
    )
    expected = subject.digest(tmp_path / "ROOT.json")
    subject.anchor(tmp_path, "ROOT.json", expected)
    payload.write_bytes(b"altered")
    with pytest.raises(subject.CheckFailure) as caught:
        subject.anchor(tmp_path, "ROOT.json", expected)
    assert caught.value.check_id == "external_anchor"


def test_raw_integrity_mutation_changes_one_coefficient_byte_only(tmp_path):
    path = tmp_path / "model.npz"
    values = {"param_linear": np.arange(6, dtype=np.float64).reshape(2, 3)}
    metadata = subject.rewrite_archive(path, {"report": {}}, values)
    original = values["param_linear"].tobytes()
    layout = {"paths": {"public_model": "model.npz"}}
    subject.mutate("PM01", tmp_path, layout, {})
    actual_metadata, actual = subject.archive(path)
    assert actual_metadata == metadata
    assert (
        sum(
            a != b
            for a, b in zip(original, actual["param_linear"].tobytes(), strict=True)
        )
        == 1
    )
    from glassbox._learner_arrays import load_arrays

    with pytest.raises(ValueError):
        load_arrays(path)


def test_added_cache_mutation_preserves_original384_and_repairs_fingerprint(tmp_path):
    path = tmp_path / "model.npz"
    values = {
        "train_future_inputs": np.arange(386 * 4, dtype=np.float64).reshape(386, 2, 2)
    }
    subject.rewrite_archive(path, {"report": {}}, values)
    subject.mutate("PM03", tmp_path, {"paths": {"public_model": "model.npz"}}, {})
    from glassbox._learner_arrays import load_arrays

    _, changed = load_arrays(path)
    np.testing.assert_array_equal(
        values["train_future_inputs"][:384], changed["train_future_inputs"][:384]
    )
    differences = np.argwhere(
        values["train_future_inputs"] != changed["train_future_inputs"]
    )
    assert differences.tolist() == [[384, 0, 0]]


def test_repair_order_is_verified_instead_of_silently_accepting_stale_seals(
    tmp_path, monkeypatch
):
    payload = tmp_path / "payload"
    payload.write_bytes(b"a")
    subject.write(tmp_path / "child.json", {"sha256": subject.digest(payload)})
    subject.write(
        tmp_path / "root.json", {"files": subject.inventory(tmp_path, ("root.json",))}
    )
    layout = {
        "mirrors": {"PM09": []},
        "reseal_order": [
            {
                "kind": "hash_links",
                "path": "child.json",
                "links": [{"pointer": ["sha256"], "payload": "payload"}],
            },
            {
                "kind": "inventory",
                "path": "root.json",
                "root": ".",
                "pointer": ["files"],
                "excluding": ["root.json"],
            },
        ],
    }
    # This contract only exercises JSON seals; model archive validation remains
    # separately covered by actual clean checks and raw/coherent archive tests.
    monkeypatch.setattr(subject, "location", lambda *args: payload)
    import glassbox._learner_arrays as arrays

    monkeypatch.setattr(arrays, "load_arrays", lambda path: ({}, {}))
    payload.write_bytes(b"b")
    subject.repair(tmp_path, layout, "PM09", {})
    subject.verify_internal(tmp_path, layout)
    layout["reseal_order"].reverse()
    payload.write_bytes(b"c")
    with pytest.raises(subject.CheckFailure) as caught:
        subject.repair(tmp_path, layout, "PM09", {})
    assert caught.value.check_id == "internal_seals"


def test_boundary_paths_reject_escape_and_symlinks(tmp_path):
    with pytest.raises(subject.CheckFailure):
        subject.under(tmp_path, "../outside")
    (tmp_path / "real").write_bytes(b"x")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(subject.CheckFailure):
        subject.inventory(tmp_path)


def test_consumer_has_precisely_the_frozen_public_import_boundary():
    import ast

    tree = ast.parse(Path(subject.__file__).read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "consumer_worker"
    )
    assignment = next(
        n
        for n in function.body
        if isinstance(n, ast.Assign) and n.targets[0].id == "allowed"
    )
    assert ast.literal_eval(assignment.value) == {
        "glassbox",
        "glassbox.learner",
        "glassbox._sequence_model",
        "glassbox._learner_arrays",
        "glassbox.recordings",
        "glassbox.io",
        "glassbox.io.recordings",
    }


def test_forecast_parity_uses_normalized_units_not_large_physical_offset():
    from glassbox.experimental.public_v4_numerics import TOLERANCES, element_check

    actual = np.array([[1000.0 + 1e-9]], dtype=np.float64)
    expected = np.array([[1000.0]], dtype=np.float64)
    norms = {"state_mean": np.array([1000.0]), "state_scale": np.array([1e-3])}
    assert element_check(actual, expected, TOLERANCES["forecast64"])["passed"]
    assert not subject.normalized_forecast_check(actual, expected, norms)["passed"]
