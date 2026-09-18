"""Frozen public-API migration checks; capture fits only the declared toy fixture.

Run this file directly with the old checkout on PYTHONPATH to capture the
baseline. All other commands use the migrated public API. Historical forecast
replay never fits a model or changes an old benchmark decision.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import io
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "docs/harness/generic-public-api-v1.json"
MANIFEST_SHA256 = "a6c966dbc5940121b187aaf7552660f1e0293a9bcaec0f12ddfcc098ea49c77f"
FIXTURE_MODELS = (
    "fitted.npz",
    "fitted-roundtrip.npz",
    "updated.npz",
    "updated-roundtrip.npz",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def frozen_plan():
    raw = MANIFEST_PATH.read_bytes()
    if sha256(raw) != MANIFEST_SHA256:
        raise ValueError("migration manifest differs from the frozen contract")
    return json.loads(raw), raw


def _frozen_ast_dump(node):
    """Serialize parsed source in the compact Python 3.13 format of the pins.

    Python 3.12's ``ast.dump`` includes empty list fields; Python 3.13 omits
    them by default. Walk the AST fields instead of rewriting dump text, which
    could also change string constants containing text such as ``args=[]``.
    Empty source containers still retain their List/Tuple/Dict node, and all
    nonempty fields and required None constants remain represented.
    """
    if isinstance(node, ast.AST):
        fields = []
        for name, value in ast.iter_fields(node):
            if value is None and getattr(type(node), name, ...) is None:
                continue
            if isinstance(value, list) and not value:
                continue
            fields.append(f"{name}={_frozen_ast_dump(value)}")
        return f"{type(node).__name__}({', '.join(fields)})"
    if isinstance(node, list):
        return f"[{', '.join(_frozen_ast_dump(value) for value in node)}]"
    return repr(node)


def normalized_source(source, contract):
    """Reverse only frozen import/doc-reference moves, preserving the rest of AST."""

    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for change in contract["doc_reference_relocations"]:
                    node.value = node.value.replace(change["new"], change["old"])
            return node

        def visit_Import(self, node):
            node.names.sort(key=lambda alias: (alias.name, alias.asname or ""))
            return node

        def visit_ImportFrom(self, node):
            for change in contract["import_relocations"]:
                if (node.module, node.level) == (
                    change["new_module"],
                    change["new_level"],
                ):
                    node.module, node.level = change["old_module"], change["old_level"]
                    break
            node.names.sort(key=lambda alias: (alias.name, alias.asname or ""))
            return node

        def generic_visit(self, node):
            node = super().generic_visit(node)
            for _field, value in ast.iter_fields(node):
                if not isinstance(value, list):
                    continue
                start = 0
                while start < len(value):
                    if not isinstance(value[start], (ast.Import, ast.ImportFrom)):
                        start += 1
                        continue
                    end = start + 1
                    while end < len(value) and isinstance(
                        value[end], (ast.Import, ast.ImportFrom)
                    ):
                        end += 1
                    value[start:end] = sorted(
                        value[start:end],
                        key=_frozen_ast_dump,
                    )
                    start = end
            return node

    tree = Normalize().visit(ast.parse(source))
    return sha256(_frozen_ast_dump(tree).encode())


def qualification_source_matches(root, name, expected):
    """Recognize only the three frozen qualification import migrations."""
    plan, _raw = frozen_plan()
    contract = plan["qualification_compatibility"]["files"].get(name)
    if contract is None or contract["original_sha256"] != expected:
        return False
    return (
        normalized_source((Path(root) / name).read_text(), contract)
        == contract["normalized_original_ast_sha256"]
    )


def archive_values(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def same_arrays(actual, expected, *, label):
    if set(actual) != set(expected):
        raise ValueError(f"{label}: array names differ")
    for name, value in actual.items():
        other = expected[name]
        if value.dtype != other.dtype or value.shape != other.shape:
            raise ValueError(f"{label}/{name}: array dtype or shape differs")
        if not np.array_equal(value, other):
            raise ValueError(f"{label}/{name}: array values differ")


def same_models(actual_path, expected_path):
    actual, expected = archive_values(actual_path), archive_values(expected_path)
    if json.loads(str(actual.pop("metadata"))) != json.loads(
        str(expected.pop("metadata"))
    ):
        raise ValueError("model metadata differs")
    same_arrays(actual, expected, label="model")


def collection_fixture(api, spec, names):
    segments = []
    for item in names:
        rng = np.random.default_rng(item["seed"])
        commands = rng.normal(size=(79, 1))
        states = np.zeros((80, 2))
        for index, command in enumerate(commands):
            states[index + 1, 0] = 0.95 * states[index, 0] + 0.08 * np.tanh(command[0])
            states[index + 1, 1] = 0.91 * states[index, 1] + 0.06 * command[0]
        segments.append(
            api.SequenceSegment(
                item["recording_id"], item["segment_id"], states, commands, spec["dt_s"]
            )
        )
    return api.SequenceCollection(
        tuple(segments),
        configuration_id=spec["configuration_id"],
        state_channels=tuple(spec["state_channels"]),
        input_channels=tuple(spec["input_channels"]),
    )


def fixture_predictions(model, batch):
    result = {}
    for name, past, commands, future in (
        ("batch", batch.past_states, batch.past_inputs, batch.future_inputs),
        ("single", batch.past_states[0], batch.past_inputs[0], batch.future_inputs[0]),
        ("prefix", batch.past_states, batch.past_inputs, batch.future_inputs[:, :2]),
    ):
        result[name] = np.asarray(model.predict(past, commands, future))
    result["envelope"] = model.envelope()
    return result


def environment():
    import jax

    return dict(
        python=sys.version,
        platform=platform.platform(),
        jax=jax.__version__,
        numpy=np.__version__,
        x64=True,
    )


def seal(output, report, plan_raw):
    (output / "manifest.json").write_bytes(plan_raw)
    write_json(output / "report.json", report)
    write_json(
        output / "files.json",
        {
            path.relative_to(output).as_posix(): sha256(path.read_bytes())
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "files.json"
        },
    )


def capture(output, implementation):
    """Fit/update the single frozen fixture, after the manifest is committed."""
    from types import SimpleNamespace

    import jax

    import glassbox

    plan, raw = frozen_plan()
    output = Path(output)
    source_root = Path(glassbox.__file__).resolve().parents[2]
    if implementation == "baseline":
        for name, item in plan["source_moves"].items():
            if sha256((source_root / name).read_bytes()) != item["baseline_sha256"]:
                raise ValueError(f"baseline learner source changed: {name}")
        learner = importlib.import_module("glassbox.experimental.default_model")
        recordings = importlib.import_module(
            "glassbox.experimental.sequence_collection"
        )
        api = SimpleNamespace(
            fit=learner.fit,
            LearnedDynamics=learner.LearnedDynamics,
            SequenceCollection=recordings.SequenceCollection,
            SequenceSegment=recordings.SequenceSegment,
        )
    else:
        api = glassbox
    output.mkdir(parents=True, exist_ok=False)
    spec = plan["fixture"]
    with jax.enable_x64(True):
        fitted = api.fit(collection_fixture(api, spec, spec["fit_recordings"]))
        if fitted.recipe != plan["recipe"]:
            raise ValueError("fixture fitted another recipe")
        fitted.save(output / "fitted.npz")
        restored = api.LearnedDynamics.load(output / "fitted.npz")
        restored.save(output / "fitted-roundtrip.npz")
        same_models(output / "fitted-roundtrip.npz", output / "fitted.npz")
        original = fitted.fingerprint()
        batch = fitted._development.batch
        predictions = fixture_predictions(fitted, batch)
        same_arrays(
            fixture_predictions(restored, batch), predictions, label="roundtrip"
        )
        updated = restored.update(
            collection_fixture(api, spec, [spec["update_recording"]])
        )
        if fitted.fingerprint() != original or restored.fingerprint() != original:
            raise ValueError("update mutated its predecessor")
        if updated.report["previous_revision"] != original:
            raise ValueError("update lost its predecessor identity")
        if updated._development.keys != fitted._development.keys:
            raise ValueError("update changed the reserved windows")
        updated.save(output / "updated.npz")
        loaded = api.LearnedDynamics.load(output / "updated.npz")
        loaded.save(output / "updated-roundtrip.npz")
        same_models(output / "updated-roundtrip.npz", output / "updated.npz")
        predictions.update(
            {
                f"updated_{key}": value
                for key, value in fixture_predictions(updated, batch).items()
            }
        )
        np.savez_compressed(output / "predictions.npz", **predictions)
    report = dict(
        kind="fixture",
        implementation=implementation,
        environment=environment(),
        fitted_fingerprint=original,
        updated_fingerprint=updated.fingerprint(),
        model_files=list(FIXTURE_MODELS),
    )
    seal(output, report, raw)
    return report


def checked_run(directory):
    directory = Path(directory)
    plan, raw = frozen_plan()
    if (directory / "manifest.json").read_bytes() != raw:
        raise ValueError("saved migration manifest changed")
    report = json.loads((directory / "report.json").read_text())
    if not isinstance(report, dict):
        raise ValueError("saved migration report is not an object")
    required = {"manifest.json", "report.json"}
    if report.get("kind") == "fixture":
        if set(report) != {
            "kind",
            "implementation",
            "environment",
            "fitted_fingerprint",
            "updated_fingerprint",
            "model_files",
        }:
            raise ValueError("saved fixture report fields changed")
        if report["implementation"] not in ("baseline", "public"):
            raise ValueError("saved fixture implementation is undeclared")
        if report["model_files"] != list(FIXTURE_MODELS):
            raise ValueError("saved fixture model file roster changed")
        required.update((*FIXTURE_MODELS, "predictions.npz"))
    elif report.get("kind") == "replay":
        if set(report) != {"kind", "accepted", "environment", "models", "forecasts"}:
            raise ValueError("saved replay report fields changed")
        count = plan["saved_evidence"]["counts"]
        if report["accepted"] is not True or report["models"] != count["models"]:
            raise ValueError("saved replay decision or model count changed")
        if (
            not isinstance(report["forecasts"], list)
            or len(report["forecasts"]) != count["forecasts"]
        ):
            raise ValueError("saved replay forecast roster changed")
        required.update(
            f"prediction-{index:03d}.npz" for index in range(1, count["forecasts"] + 1)
        )
    else:
        raise ValueError("unknown migration run kind")
    declared_environment = report["environment"]
    if (
        not isinstance(declared_environment, dict)
        or set(declared_environment) != {"python", "platform", "jax", "numpy", "x64"}
        or declared_environment["x64"] is not True
        or any(
            not isinstance(declared_environment[key], str)
            or not declared_environment[key]
            for key in ("python", "platform", "jax", "numpy")
        )
    ):
        raise ValueError("saved migration environment fields changed")
    inventory = json.loads((directory / "files.json").read_text())
    actual_names = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "files.json"
    }
    if set(inventory) != actual_names or actual_names != required:
        raise ValueError("saved migration file set changed")
    for name, expected in inventory.items():
        if sha256((directory / name).read_bytes()) != expected:
            raise ValueError(f"saved migration artifact changed: {name}")
    return report


def compare(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    old, new = checked_run(baseline), checked_run(candidate)
    if old["kind"] != "fixture" or new["kind"] != "fixture":
        raise ValueError("comparison requires two fixture captures")
    if old["implementation"] != "baseline" or new["implementation"] != "public":
        raise ValueError("comparison requires the baseline and public implementations")
    if old["environment"] != new["environment"]:
        raise ValueError("fixture environments differ")
    verify(baseline)
    verify(candidate)
    for name in ("fitted_fingerprint", "updated_fingerprint", "model_files"):
        if old[name] != new[name]:
            raise ValueError(f"fixture semantic equality failed: {name}")
    for name in FIXTURE_MODELS:
        same_models(candidate / name, baseline / name)
    same_arrays(
        archive_values(candidate / "predictions.npz"),
        archive_values(baseline / "predictions.npz"),
        label="fixture forecasts",
    )
    return dict(
        accepted=True, meaning="Exact same-environment fit/update/artifact parity."
    )


def public_contract():
    plan, _raw = frozen_plan()
    code = """
import inspect,json,pkgutil,sys
import glassbox
contract = json.loads(sys.argv[1])
exports = contract["exports"]
assert glassbox.__all__ == exports
assert len(exports) == len(set(exports))
assert all(hasattr(glassbox,name) for name in exports)
assert set(exports).isdisjoint(info.name for info in pkgutil.iter_modules(glassbox.__path__))
for dotted, expected in contract["signatures"].items():
    target = glassbox
    for name in dotted.split("."):
        target = getattr(target, name)
    actual = []
    for parameter in inspect.signature(target).parameters.values():
        assert parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        if parameter.default is parameter.empty:
            actual.append(parameter.name)
        else:
            assert parameter.default is None
            actual.append(parameter.name + "=None")
    assert actual == expected, (dotted, actual, expected)
for name in contract["removed"]:
    assert not hasattr(glassbox,name), name
for prefix in contract["deferred"]:
    assert not any(name == prefix or name.startswith(prefix+".") for name in sys.modules), prefix
print(json.dumps({"accepted":True,"exports":exports}))
"""
    contract = dict(
        exports=plan["public_api"]["exports"],
        signatures=plan["public_api"]["signatures"],
        removed=plan["public_api"]["removed_root_exports"],
        deferred=[
            "glassbox." + name
            for name in (
                "experimental",
                "belief",
                "fitting",
                "control",
                "io",
                "integrations",
                "workflows",
                "cli",
                "core.dynamics",
                "core.model",
                "core.identification",
            )
        ],
    )
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(contract)],
        check=True,
        capture_output=True,
        text=True,
    )
    for name in plan["public_api"]["removed_paths"]:
        if (ROOT / name).exists():
            raise ValueError(f"replaced path remains: {name}")
    return json.loads(result.stdout)


def checked_evidence(root, plan):
    root = Path(root)
    for name, expected in plan["saved_evidence"]["sha256"].items():
        if sha256((root / name).read_bytes()) != expected:
            raise ValueError(f"frozen migration evidence changed: {name}")


def replay_models(artifacts, output=None):
    import jax

    from glassbox import LearnedDynamics
    from glassbox.experimental.harness import replay

    plan, raw = frozen_plan()
    artifacts = Path(artifacts)
    checked_evidence(artifacts, plan)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
    rows = []
    with jax.enable_x64(True):
        for item in plan["saved_evidence"]["models"]:
            learned = LearnedDynamics.load(artifacts / item["path"])
            if (
                learned.fingerprint() != item["fingerprint"]
                or learned.recipe != plan["recipe"]
            ):
                raise ValueError(f"saved model identity changed: {item['path']}")
            saved = io.BytesIO()
            learned.save(saved)
            same_models(io.BytesIO(saved.getvalue()), artifacts / item["path"])
            for forecast in item["forecasts"]:
                data = archive_values(artifacts / forecast["path"])
                args = [
                    data[name]
                    for name in ("past_states", "past_inputs", "future_inputs")
                ]
                predicted = np.asarray(learned.predict(*args))
                expected = data[forecast["prediction_key"]]
                if (
                    list(predicted.shape) != forecast["shape"]
                    or not np.isfinite(predicted).all()
                ):
                    raise ValueError("saved forecast shape or finiteness changed")
                np.testing.assert_allclose(
                    predicted,
                    expected,
                    **plan["saved_evidence"]["prediction_tolerance"],
                )
                np.testing.assert_allclose(
                    replay(learned._model, *args),
                    expected,
                    **plan["saved_evidence"]["prediction_tolerance"],
                )
                if forecast["envelope_key"] is not None:
                    envelope = np.broadcast_to(
                        learned.envelope(predicted.shape[-2]), predicted.shape
                    )
                    np.testing.assert_array_equal(
                        envelope, data[forecast["envelope_key"]]
                    )
                row = dict(
                    path=forecast["path"],
                    model=item["path"],
                    fingerprint=learned.fingerprint(),
                    maximum_difference=float(np.max(np.abs(predicted - expected))),
                )
                rows.append(row)
                if output is not None:
                    np.savez_compressed(
                        output / f"prediction-{len(rows):03d}.npz", prediction=predicted
                    )
    report = dict(
        kind="replay",
        accepted=True,
        environment=environment(),
        models=len(plan["saved_evidence"]["models"]),
        forecasts=rows,
    )
    if output is not None:
        seal(output, report, raw)
    return report


def verify(directory, artifacts=None):
    """Recompute saved migration predictions; never refit a fixture or real model."""
    import jax

    from glassbox import LearnedDynamics

    directory = Path(directory)
    report = checked_run(directory)
    if report["kind"] == "replay":
        if artifacts is None:
            raise ValueError("replay verification requires the pinned artifact root")
        fresh = replay_models(artifacts)
        if report != fresh:
            raise ValueError("saved replay summary changed")
        with jax.enable_x64(True):
            for index, row in enumerate(fresh["forecasts"], 1):
                model = LearnedDynamics.load(Path(artifacts) / row["model"])
                data = archive_values(Path(artifacts) / row["path"])
                predicted = np.asarray(
                    model.predict(
                        *[
                            data[name]
                            for name in ("past_states", "past_inputs", "future_inputs")
                        ]
                    )
                )
                stored = archive_values(directory / f"prediction-{index:03d}.npz")
                same_arrays(
                    stored, {"prediction": predicted}, label="saved public replay"
                )
        return fresh
    if report["kind"] != "fixture":
        raise ValueError("unknown migration run kind")
    if report["environment"] != environment():
        raise ValueError("fixture verification environment differs from its capture")
    with jax.enable_x64(True):
        fitted = LearnedDynamics.load(directory / "fitted.npz")
        updated = LearnedDynamics.load(directory / "updated.npz")
        if (
            fitted.fingerprint() != report["fitted_fingerprint"]
            or updated.fingerprint() != report["updated_fingerprint"]
        ):
            raise ValueError("saved fixture identity changed")
        if updated.report["previous_revision"] != fitted.fingerprint():
            raise ValueError("saved update predecessor changed")
        for name in ("fitted", "updated"):
            same_models(directory / f"{name}.npz", directory / f"{name}-roundtrip.npz")
        expected = fixture_predictions(fitted, fitted._development.batch)
        expected.update(
            {
                f"updated_{key}": value
                for key, value in fixture_predictions(
                    updated, fitted._development.batch
                ).items()
            }
        )
        same_arrays(
            archive_values(directory / "predictions.npz"),
            expected,
            label="saved fixture forecasts",
        )
    return dict(accepted=True, kind="fixture", no_fit=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("capture")
    command.add_argument(
        "--implementation", choices=("baseline", "public"), required=True
    )
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("compare")
    command.add_argument("--baseline", type=Path, required=True)
    command.add_argument("--candidate", type=Path, required=True)
    command = commands.add_parser("replay")
    command.add_argument("--artifacts", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("verify")
    command.add_argument("directory", type=Path)
    command.add_argument("--artifacts", type=Path)
    commands.add_parser("public")
    args = parser.parse_args(argv)
    if args.command == "capture":
        result = capture(args.output, args.implementation)
    elif args.command == "compare":
        result = compare(args.baseline, args.candidate)
    elif args.command == "replay":
        result = replay_models(args.artifacts, args.output)
    elif args.command == "verify":
        result = verify(args.directory, args.artifacts)
    else:
        result = public_contract()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
