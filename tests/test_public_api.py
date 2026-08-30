import json
import subprocess
import sys

import glassbox


def test_public_api_exports_resolve_and_are_unique() -> None:
    assert len(glassbox.__all__) == len(set(glassbox.__all__))
    assert all(hasattr(glassbox, name) for name in glassbox.__all__)


def test_importing_public_api_does_not_load_workflow_modules() -> None:
    code = """
import json
import sys
import glassbox

workflow_modules = {
    "glassbox.fit_cli",
    "glassbox.fixedwing_gate",
    "glassbox.policy_selection",
    "glassbox.profile_benchmark",
    "glassbox.predictive_ensemble",
    "glassbox.nmpc_benchmark",
    "glassbox.source_group_benchmark",
}
print(json.dumps(sorted(workflow_modules.intersection(sys.modules))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []
