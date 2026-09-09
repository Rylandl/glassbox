"""The static ``glassbox`` subcommand tree.

Every leaf names a module and an entry point as ``"module:function"`` but does
not import it: ``glassbox --help`` has to render the whole tree in an
environment where no optional extra is installed, so summaries live here as
plain strings and modules are imported only when a leaf is actually dispatched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """One dispatchable leaf of the ``glassbox`` command tree."""

    name: str
    target: str
    summary: str
    extra: str | None = None
    subcommands: tuple[str, ...] = ()


TREE: tuple[Command, ...] = (
    Command(
        name="extract",
        target="glassbox.cli.extract:main",
        summary="convert PX4 ULogs to canonical trajectory NPZ files",
        extra="px4",
    ),
    Command(
        name="corpus",
        target="glassbox.cli.corpus:main",
        summary="list, fetch, and prepare the pinned reference corpora",
        subcommands=("list", "fetch", "prepare"),
    ),
    Command(
        name="synthetic",
        target="glassbox.cli.synthetic:main",
        summary="generate canonical synthetic trajectories for either family",
    ),
    Command(
        name="fit",
        target="glassbox.cli.fit:main",
        summary="fit a dynamics belief and report from trajectory NPZ files",
    ),
    Command(
        name="evaluate",
        target="glassbox.cli.evaluate:main",
        summary="score models on held-out flight under one named protocol",
    ),
    Command(
        name="benchmark",
        target="glassbox.cli.benchmark:main",
        summary="run one maintained closed-loop or corpus benchmark",
        subcommands=("nmpc", "recovery", "cascade-x8"),
    ),
    Command(
        name="record-results",
        target="glassbox.cli.record_results:main",
        summary="regenerate the recorded artifacts under docs/results/",
    ),
    Command(
        name="sitl-profile",
        target="glassbox.cli.sitl_profile:main",
        summary="fly one bounded PX4 SITL maneuver profile over MAVLink",
        extra="px4",
    ),
    Command(
        name="px4-shadow",
        target="glassbox.cli.px4_shadow:main",
        summary="passive NMPC shadow against live PX4 telemetry; never transmits",
        extra="px4",
    ),
)


def leaf_paths(nodes: tuple[Command, ...] = TREE) -> list[tuple[str, ...]]:
    """Return the argv prefix of every dispatchable leaf, in tree order."""

    return [(node.name,) for node in nodes]


def find(path: tuple[str, ...]) -> Command | None:
    """Return the leaf at ``path``, or ``None`` when the path is unknown.

    Every leaf sits at the top level; a command's own subcommands are parsed by
    the command, so any longer path is unknown here.
    """

    if len(path) != 1:
        return None
    return next((item for item in TREE if item.name == path[0]), None)
