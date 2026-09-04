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
        name="fit",
        target="glassbox.cli.fit:main",
        summary="fit a dynamics belief and report from trajectory NPZ files",
    ),
    Command(
        name="synthetic",
        target="glassbox.cli.synthetic_demo:main",
        summary="run the synthetic multirotor parameter-recovery demonstration",
    ),
    Command(
        name="fixedwing-synthetic",
        target="glassbox.cli.fixedwing:main",
        summary="generate canonical synthetic fixed-wing trajectories",
    ),
    Command(
        name="profile-benchmark",
        target="glassbox.workflows.holdout:profile_main",
        summary="run leave-one-maneuver-profile-out dynamics identification",
    ),
    Command(
        name="source-benchmark",
        target="glassbox.workflows.holdout:source_group_main",
        summary="run leave-one-source-group-out dynamics identification",
    ),
    Command(
        name="adaptive-recovery",
        target="glassbox.workflows.adaptive_recovery_benchmark:main",
        summary="prewarmed synthetic recovery after a configuration change",
    ),
    Command(
        name="nmpc-benchmark",
        target="glassbox.workflows.nmpc_benchmark:main",
        summary="maintained closed-loop NMPC acceptance and timing benchmark",
    ),
    Command(
        name="record-results",
        target="glassbox.workflows.record_results:main",
        summary="regenerate the recorded artifacts under docs/results/",
    ),
    Command(
        name="sitl-profile",
        target="glassbox.io.sitl_profile:main",
        summary="fly bounded PX4 SITL position/yaw profiles over MAVLink",
        extra="px4",
    ),
    Command(
        name="fixedwing-sitl-profile",
        target="glassbox.io.fixedwing_sitl_profile:main",
        summary="fly bounded PX4 fixed-wing attitude/throttle profiles",
        extra="px4",
    ),
    Command(
        name="px4-nmpc-shadow",
        target="glassbox.integrations.px4_nmpc_shadow:main",
        summary="passive NMPC shadow against live PX4 telemetry; never transmits",
        extra="px4",
    ),
    Command(
        name="ulog",
        target="glassbox.cli.ulog:main",
        summary="inspect PX4 ULogs and prepare the ARP and IDF-DS corpora",
        extra="px4",
        subcommands=(
            "inspect",
            "extract",
            "extract-fixedwing",
            "prepare-arp",
            "prepare-idf",
        ),
    ),
    Command(
        name="nanodrone",
        target="glassbox.cli.nanodrone:main",
        summary="fetch, convert, and evaluate the IDSIA Nano-Quadrotor benchmark",
        subcommands=(
            "inspect",
            "extract",
            "fetch",
            "extract-dataset",
            "prepare",
            "evaluate",
        ),
    ),
    Command(
        name="x8",
        target="glassbox.cli.x8:main",
        summary="fetch, convert, and evaluate the NTNU Skywalker X8 campaign",
        subcommands=(
            "inspect",
            "extract",
            "fetch",
            "extract-dataset",
            "prepare",
            "evaluate",
            "evaluate-cascade",
            "diagnose-cascade",
        ),
    ),
    Command(
        name="epfl",
        target="glassbox.cli.epfl:main",
        summary="fetch, convert, and evaluate the EPFL TOPOPlane2 release",
        extra="ros",
        subcommands=("inspect", "extract", "fetch", "prepare", "evaluate"),
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
