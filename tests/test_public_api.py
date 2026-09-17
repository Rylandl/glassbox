"""Contracts for the stable ``glassbox`` surface and its subpackage boundary."""

from __future__ import annotations

import json
import pkgutil
import subprocess
import sys

import glassbox

# Snapshot of the stable surface, grouped in the order a reader meets it. A
# name belongs here only if the README or a docs/concepts page uses it, or if
# it is the type of a public function's argument or return value. Additions
# and removals are both deliberate edits here, so a name never enters or
# leaves the public API by accident.
EXPECTED_PUBLIC_API = (
    "fit",
    "LearnedDynamics",
    "SequenceCollection",
    "SequenceSegment",
    "segments_from_mask",
)

# Subpackages that a bare ``import glassbox`` must never pull in: workflows and
# command-line front ends are heavy, and corpus and integration adapters need
# optional extras.
DEFERRED_SUBPACKAGES = (
    "cli",
    "integrations",
    "io",
    "workflows",
    "experimental",
    "belief",
    "fitting",
    "control",
    "core",
)


def test_public_api_exports_resolve_and_are_unique() -> None:
    assert len(glassbox.__all__) == len(set(glassbox.__all__))
    assert all(hasattr(glassbox, name) for name in glassbox.__all__)


def test_public_api_matches_the_recorded_surface() -> None:
    assert tuple(glassbox.__all__) == EXPECTED_PUBLIC_API


def test_public_api_names_never_shadow_a_submodule() -> None:
    submodules = {info.name for info in pkgutil.iter_modules(glassbox.__path__)}
    assert submodules.isdisjoint(glassbox.__all__)


def test_importing_public_api_does_not_load_deferred_subpackages() -> None:
    code = f"""
import json
import sys
import glassbox

deferred = {DEFERRED_SUBPACKAGES!r}
loaded = sorted(
    name
    for name in sys.modules
    if name.startswith("glassbox.") and name.split(".")[1] in deferred
)
print(json.dumps(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_public_fit_is_the_one_recipe_without_a_selector():
    import inspect

    from glassbox.learner import LearnedDynamics, fit

    assert glassbox.fit is fit
    assert glassbox.LearnedDynamics is LearnedDynamics
    assert list(inspect.signature(glassbox.fit).parameters) == ["recordings"]
    assert list(inspect.signature(glassbox.LearnedDynamics.update).parameters) == [
        "self",
        "recordings",
    ]


def test_replaced_root_names_and_module_paths_are_removed():
    import importlib.util

    for name in (
        "FitSpec",
        "DynamicsBelief",
        "FitOutcome",
        "DynamicsParams",
        "NMPCController",
    ):
        assert not hasattr(glassbox, name)
    for name in (
        "default_model",
        "sequence_model",
        "sequence_collection",
        "arrays",
        "sequence_diagnostics",
    ):
        assert importlib.util.find_spec(f"glassbox.experimental.{name}") is None
