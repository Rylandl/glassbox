"""Pure source-association checks; no saved-model prediction or fitting."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

DRIVER = (
    Path(__file__).resolve().parents[1]
    / "docs/diagnostics/public-mean-corrected-api-replay.py"
)
spec = importlib.util.spec_from_file_location("_api_replay", DRIVER)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
CORE = "src/glassbox/_sequence_model.py"


@pytest.fixture
def associated(tmp_path):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    original = "def fit():\n    return 1\nclass SequenceModel:\n    def rollout(self, x):\n        return x\n    def memory_state(self, x):\n        return x\n"
    current = original.replace("return x", "return x + 0")
    for root, text in ((old_root, original), (new_root, current)):
        (root / CORE).parent.mkdir(parents=True)
        (root / CORE).write_text(text)
    common = dict(
        format="glassbox-public-mean-implementation-v1",
        protocol_sha256="protocol",
        runtime={"numpy": "pinned"},
        machine="arm64",
        interpreter_sha256="python",
        oracle_commit="oracle",
        oracle_source_sha256={"old": "source"},
        consumer_source_sha256={"consumer": "source"},
    )
    old = dict(
        common,
        public_root=str(old_root),
        implementation_commit="old",
        public_source_sha256={CORE: "old-core", "src/glassbox/learner.py": "same"},
    )
    new = dict(
        copy.deepcopy(common),
        public_root=str(new_root),
        implementation_commit="new",
        public_source_sha256={CORE: "new-core", "src/glassbox/learner.py": "same"},
    )
    policy = {
        "allowed_common_source_differences": {
            CORE: {"old": "old-core", "current": "new-core"}
        }
    }
    return old, new, policy


def test_only_declared_inference_bodies_may_differ(associated):
    result = bridge.source_association(*associated)
    assert result["unchanged_core_outside_two_inference_methods"]
    assert result["old_commit"] == "old" and result["new_commit"] == "new"


@pytest.mark.parametrize(
    "change", ["other_source", "missing_source", "runtime", "fitter", "signature"]
)
def test_source_bridge_rejects_unapproved_differences(associated, change):
    old, new, policy = associated
    if change == "other_source":
        new["public_source_sha256"]["src/glassbox/learner.py"] = "changed"
    elif change == "missing_source":
        del new["public_source_sha256"]["src/glassbox/learner.py"]
    elif change == "runtime":
        new["runtime"]["numpy"] = "different"
    else:
        path = Path(new["public_root"]) / CORE
        path.write_text(
            path.read_text().replace("return 1", "return 2")
            if change == "fitter"
            else path.read_text().replace(
                "rollout(self, x)", "rollout(self, x, option=False)"
            )
        )
    with pytest.raises(AssertionError):
        bridge.source_association(old, new, policy)


def test_check_comparison_ignores_only_container_hash():
    original = [
        {
            "check": "coherent_defect",
            "sha256": "old-zip",
            "status": "passed",
            "payload_fingerprint": "same",
        }
    ]
    fresh = copy.deepcopy(original)
    fresh[0]["sha256"] = "new-zip"
    assert bridge.normalized_checks(original) == bridge.normalized_checks(fresh)
    fresh[0]["status"] = "failed"
    assert bridge.normalized_checks(original) != bridge.normalized_checks(fresh)


@pytest.mark.parametrize("defect", [None, "blob", "revision"])
def test_moved_historical_root_uses_only_pinned_commit_blobs(
    tmp_path, monkeypatch, defect
):
    revision = "a" * 40
    source = b"def fit():\n    return 1\n"
    root = tmp_path / "advanced-checkout"
    (root / CORE).parent.mkdir(parents=True)
    (root / CORE).write_bytes(b"not the old source")
    binding = tmp_path / "old-binding.json"
    binding.write_text(
        json.dumps(
            dict(
                public_root=str(root),
                implementation_commit=revision,
                public_source_sha256={CORE: hashlib.sha256(source).hexdigest()},
            )
        )
    )
    entry = dict(
        path=str(binding), sha256=bridge.sha(binding), original_commit=revision
    )
    seen = []

    def git(command, **kwargs):
        seen.append(command)
        assert command[:3] == ["git", "-C", str(root)]
        if command[3] == "rev-parse":
            assert command[4] == revision + "^{commit}"
            return ("b" * 40 if defect == "revision" else revision) + "\n"
        assert command[3:] == ["show", revision + ":" + CORE]
        return b"wrong blob" if defect == "blob" else source

    monkeypatch.setattr(bridge.subprocess, "check_output", git)
    if defect:
        with pytest.raises(AssertionError):
            bridge.historical_binding(entry)
    else:
        old, core = bridge.historical_binding(entry)
        assert old["implementation_commit"] == revision and core == source
        assert len(seen) == 2
