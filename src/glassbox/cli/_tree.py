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


@dataclass(frozen=True)
class Group:
    """A namespace whose children are separate entry points."""

    name: str
    summary: str
    commands: tuple[Command, ...]


Node = Command | Group

TREE: tuple[Node, ...] = (
    Command(
        name="fit",
        target="glassbox.cli.fit:main",
        summary="fit a dynamics belief and report from trajectory NPZ files",
    ),
    Command(
        name="prior",
        target="glassbox.cli.prior:main",
        summary="build one explicit fleet/configuration prior from fitted beliefs",
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
        target="glassbox.workflows.profile_benchmark:main",
        summary="run leave-one-maneuver-profile-out dynamics identification",
    ),
    Command(
        name="source-benchmark",
        target="glassbox.workflows.source_group_benchmark:main",
        summary="run leave-one-source-group-out dynamics identification",
    ),
    Command(
        name="ensemble-benchmark",
        target="glassbox.workflows.predictive_ensemble:main",
        summary="offline grouped predictive ensembles and uncertainty diagnostics",
    ),
    Command(
        name="select-policy",
        target="glassbox.workflows.policy_selection:main",
        summary="select a shared fitting policy across platforms and profiles",
    ),
    Command(
        name="adaptation-benchmark",
        target="glassbox.workflows.adaptation_benchmark:main",
        summary="compact synthetic evidence for fleet-prior live adaptation",
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
        name="fixedwing-gate",
        target="glassbox.workflows.fixedwing_gate:main",
        summary="cross-airframe fixed-wing development and promotion gate",
        subcommands=("evaluate", "compare", "screen"),
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
    Group(
        name="crazyflow",
        summary="Crazyflow hidden-plant simulation diagnostics",
        commands=(
            Command(
                name="prototype",
                target="glassbox.integrations.crazyflow_prototype:main",
                summary="fixed hidden-plant prototype for adjustable-arm recovery",
                extra="crazyflow",
            ),
            Command(
                name="bootstrap",
                target="glassbox.integrations.crazyflow_bootstrap:main",
                summary="no-airframe-prior bootstrap identification",
                extra="crazyflow",
            ),
            Command(
                name="throw",
                target="glassbox.integrations.crazyflow_throw:main",
                summary="unpowered throw, online identification, and arrest",
                extra="crazyflow",
            ),
            Command(
                name="animation",
                target="glassbox.integrations.crazyflow_animation:main",
                summary="render the bootstrap diagnostic as an annotated video",
                extra="crazyflow-animation",
            ),
            Command(
                name="throw-animation",
                target="glassbox.integrations.crazyflow_animation:throw_main",
                summary="render the throw diagnostic as an annotated video",
                extra="crazyflow-animation",
            ),
        ),
    ),
)


def leaf_paths(nodes: tuple[Node, ...] = TREE) -> list[tuple[str, ...]]:
    """Return the argv prefix of every dispatchable leaf, in tree order."""

    paths: list[tuple[str, ...]] = []
    for node in nodes:
        if isinstance(node, Group):
            paths.extend((node.name, *path) for path in leaf_paths(node.commands))
        else:
            paths.append((node.name,))
    return paths


def find(path: tuple[str, ...]) -> Node | None:
    """Return the node at ``path``, or ``None`` when the path is unknown."""

    nodes: tuple[Node, ...] = TREE
    node: Node | None = None
    for name in path:
        node = next((item for item in nodes if item.name == name), None)
        if node is None:
            return None
        nodes = node.commands if isinstance(node, Group) else ()
    return node
