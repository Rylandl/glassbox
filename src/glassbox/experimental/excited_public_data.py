"""Byte-exact historical excitation imports with separate current provenance."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def trusted_excitation(p, flight):
    """Verify the external source anchor and every sealed source payload."""
    declared = p["prior_excitation_bundle"]
    source = Path(declared["path"])
    if flight.digest(source / "run.json") != declared["manifest_sha256"]:
        raise ValueError("prior excitation bundle differs from trusted anchor")
    manifest = flight.verify_files(source, "run.json")
    previous = p["prior_excitation_protocol"]
    if (
        flight.digest(ROOT / previous["path"]) != previous["sha256"]
        or manifest["protocol_sha256"] != previous["sha256"]
    ):
        raise ValueError("prior excitation protocol differs from trusted source")
    return source


def _files(directory, seal, *, exact):
    """Reject unsafe seal paths and unsealed additions in an imported directory."""
    expected = set(seal["files"]) | {"seal.json"}
    for relative in expected:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe imported payload path")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("excitation payload must not contain symlinks")
        if path.is_file():
            actual.add(str(path.relative_to(directory)))
    if (exact and actual != expected) or not expected <= actual:
        raise ValueError("imported excitation file roster differs")
    return sorted(expected)


def _source_data(simulator, p, flight):
    source = trusted_excitation(p, flight)
    directory = source / simulator / "excitation"
    expected_seal = p["imported_calibration"]["excitation_seal_sha256"][simulator]
    if flight.digest(directory / "seal.json") != expected_seal:
        raise ValueError("source excitation seal differs from frozen import")
    seal = flight.verify_files(directory, "seal.json")
    files = _files(directory, seal, exact=True)
    manifest = flight.read_json(source / "run.json")
    for relative in files:
        key = str(Path(simulator) / "excitation" / relative)
        if manifest["files"].get(key) != flight.digest(directory / relative):
            raise ValueError("imported source payload is not anchored by the bundle")
    if seal["runtime"] != manifest["runtime"]:
        raise ValueError("historical excitation runtime differs from source bundle")
    return source, directory, seal, files


def _calibration(simulator, output, source, flight, p, runtime):
    current = Path(output) / simulator / "data"
    previous = source / simulator / "data"
    current_seal = flight.verify_files(current, "seal.json")
    previous_seal = flight.verify_files(previous, "seal.json")
    roles = p["planned_automatic_roles"][simulator]
    entries = [
        entry
        for entry in p["recordings"]
        if entry["simulator"] == simulator and entry["role"] == "calibration_pool"
    ]
    admitted = [entry["id"] for entry in entries]
    if (
        current_seal["runtime"] != runtime
        or len(admitted) != p["counts"]["calibration_pool_per_simulator"]
        or len(set(admitted)) != len(admitted)
        or set(admitted) != set(roles["training"] + roles["development"])
        or flight.roles_for(admitted) != roles
    ):
        raise ValueError("current calibration runtime or frozen admission differs")
    for directory, seal in ((current, current_seal), (previous, previous_seal)):
        if (
            seal["admitted"] != admitted
            or seal["roles"] != roles
            or flight.read_json(directory / "roles.json") != roles
        ):
            raise ValueError("imported and current calibration roles differ")
    if (current / "configuration.json").read_bytes() != (
        previous / "configuration.json"
    ).read_bytes():
        raise ValueError("current and historical physical configurations differ")
    records = [
        [
            r
            for r in flight.read_json(directory / "records.json")
            if r["role"] == "calibration_pool"
        ]
        for directory in (current, previous)
    ]
    if any([r["id"] for r in rows] != admitted for rows in records):
        raise ValueError("current and historical calibration record rosters differ")
    if records[0] != records[1]:
        raise ValueError("regenerated original calibration metadata differs")
    arrays_count = 0
    for entry, record in zip(entries, records[0], strict=True):
        if any(record[key] != value for key, value in entry.items()):
            raise ValueError("original calibration entry differs from frozen roster")
        for directory in (current, previous):
            if record != flight.read_json(directory / (record["prefix"] + ".json")):
                raise ValueError("original calibration record mirror differs")
        new = flight.load_arrays(current / (record["prefix"] + ".npz"))
        old = flight.load_arrays(previous / (record["prefix"] + ".npz"))
        flight.array_equal(new, old, "import/original/" + record["id"])
        arrays_count += len(new)
    return current_seal, previous_seal, arrays_count


def check_import(simulator, output, flight, p, runtime):
    """Validate imported bytes without reinterpreting their historical runtime."""
    from .command_excitation_data import _compare

    source, previous, seal, files = _source_data(simulator, p, flight)
    directory = Path(output) / simulator / "excitation"
    imported_seal = flight.verify_files(directory, "seal.json")
    if _files(directory, imported_seal, exact=True) != files:
        raise ValueError("copied excitation file inventory differs")
    for relative in files:
        if (directory / relative).read_bytes() != (previous / relative).read_bytes():
            raise ValueError(
                f"imported excitation file differs from source: {relative}"
            )
    current, original, arrays_count = _calibration(
        simulator, output, source, flight, p, runtime
    )
    if (
        seal["common_data_seal"] != flight.digest(source / simulator / "data/seal.json")
        or seal["runtime"] != original["runtime"]
    ):
        raise ValueError("historical excitation provenance link differs")
    old_protocol = flight.read_json(ROOT / p["prior_excitation_protocol"]["path"])
    # This checks current original calibration, unchanged development, all
    # prefixes and random tapes. It deliberately omits old seal/current-runtime
    # equality: that claim would be false for imported historical evidence.
    comparison = _compare(simulator, output, flight, old_protocol, sealed=False)
    if (
        imported_seal["roles"] != current["roles"]
        or imported_seal["admitted"] != current["admitted"]
    ):
        raise ValueError("imported excitation admission differs from current data")
    return {
        "kind": "byte-exact-historical-excitation-import",
        "simulator": simulator,
        "source_bundle_path": str(source),
        "source_bundle_manifest_sha256": p["prior_excitation_bundle"][
            "manifest_sha256"
        ],
        "historical_protocol_sha256": p["prior_excitation_protocol"]["sha256"],
        "imported_data_seal_sha256": flight.digest(directory / "seal.json"),
        "historical_common_data_seal_sha256": seal["common_data_seal"],
        "current_common_data_seal_sha256": flight.digest(
            Path(output) / simulator / "data/seal.json"
        ),
        "historical_runtime": seal["runtime"],
        "current_runtime": runtime,
        "imported_file_count": len(files),
        "calibration_arrays_exact": arrays_count,
        "original_calibration_parents_exact": comparison["calibration_parents"],
        "development_parent_files_exact": comparison["copied_development_parents"],
        "roles_and_admission_exact": True,
        "copied_files_and_seal_byte_exact": True,
        "current_calibration_comparison": comparison,
    }


def import_excitation(simulator, output, flight, p, runtime):
    """Copy only anchored historical files and retain the original seal bytes."""
    source, previous, _, files = _source_data(simulator, p, flight)
    _calibration(simulator, output, source, flight, p, runtime)
    directory = Path(output) / simulator / "excitation"
    directory.mkdir(parents=True, exist_ok=False)
    for relative in files:
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous / relative, target)
    return check_import(simulator, output, flight, p, runtime)


def replay_import(simulator, output, flight, p, runtime):
    """Replay original physics under its own protocol, then reverify the copy."""
    from . import command_excitation_experiment as historical
    from .command_excitation_data import replay_excitation

    before = check_import(simulator, output, flight, p, runtime)
    source = trusted_excitation(p, flight)
    if historical.PROTOCOL != ROOT / p["prior_excitation_protocol"]["path"]:
        raise ValueError("historical replay protocol location differs")
    old_protocol, old_runtime = historical.protocol(), historical.check_sources()
    if old_runtime != before["historical_runtime"]:
        raise ValueError("historical physics replay runtime/source identity differs")
    evidence = replay_excitation(
        simulator, source, historical.flight(), old_protocol, old_runtime
    )
    if (
        evidence.get("exact") is not True
        or evidence.get("regenerated_training_parents")
        != p["counts"]["excited_training_parents_per_simulator"]
        or evidence.get("copied_development_parents")
        != p["counts"]["unchanged_development_copies_per_simulator"]
        or evidence.get("calibration_parents")
        != p["counts"]["calibration_pool_per_simulator"]
        or not evidence.get("arrays")
        or evidence.get("replay_arrays") != evidence["arrays"]
    ):
        raise ValueError("historical excitation replay is incomplete")
    after = check_import(simulator, output, flight, p, runtime)
    if before != after:
        raise ValueError("imported data or source changed during physics replay")
    return {
        "reuse": after,
        "historical_excitation_replay": evidence,
        "copied_files_reverified": True,
    }
