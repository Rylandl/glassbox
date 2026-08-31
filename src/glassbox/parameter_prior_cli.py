"""Build one explicit fleet/configuration prior from fitted beliefs."""

from __future__ import annotations

import argparse
from pathlib import Path

from glassbox.belief import DynamicsBelief
from glassbox.parameter_prior import StructuredParameterPrior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "members",
        nargs="+",
        type=Path,
        help="fitted DynamicsBelief artifacts from one vehicle family",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    members = tuple(DynamicsBelief.load(path) for path in args.members)
    labels = tuple(str(path.resolve()) for path in args.members)
    prior = StructuredParameterPrior.from_beliefs(
        members,
        source="fleet_belief_artifacts",
        member_labels=labels,
    )
    prior.save(args.output)
    print(
        f"wrote {args.output}: {prior.member_count} members, "
        f"empirical rank {prior.empirical_rank}/{len(prior.parameter_names)}, "
        "natural-coordinate completion fraction "
        f"{prior.completion_fraction_in_natural_coordinates:.3f}"
    )


if __name__ == "__main__":
    main()
