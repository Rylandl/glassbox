"""Score fitted models on held-out flight under one named protocol.

Three shapes, one command:

``glassbox evaluate MODEL NPZ... --protocol P``
    score one saved belief on flights it was not fitted to. ``--model
    NAME=PATH``, repeated, scores several models on the same flights against
    the same baseline and reports each pair's comparison; the positional model
    is then omitted and every path is a trajectory.

``glassbox evaluate --hold-out KEY NPZ...``
    the leave-one-label-out runner: one fold per value of the label KEY, each
    fold fitted on the rest and scored on the held-out flights, plus the
    equal-fold aggregate and the distribution across folds. ``profile`` and
    ``source_group`` are the two labels the corpora carry. ``--limit-folds N``
    fits only the first N of them, for exercising the chain without waiting
    for a corpus-scale run; the summary records the shortened fold selection.

``glassbox evaluate --fit-reports A B``
    score models whose held-out horizon tables one fit already measured,
    against the protocol's baseline over the same flights. This is a
    same-flight characterization when the fits shared a flight, and the report
    says so.

``--protocol`` names the scoring convention: ``windowed`` is back-to-back
fixed-horizon rollouts with the physical equality floors, ``x8`` the NTNU
campaign's every-sample convention, and ``nanodrone`` the IDSIA benchmark's
per-step Euclidean error against holding the initial state. Two numbers
produced under different conventions are not comparable, so every report
carries the whole policy.

``--corpus NAME`` checks the flights against that corpus's published
evaluation split before scoring, and records its citation in the report. A
trajectory outside the published split is a different measurement, not a worse
number.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from glassbox.core.data import load_trajectory_npz
from glassbox.fitting import FitSpec, LossPolicy
from glassbox.io.corpus import REFERENCE_CORPORA
from glassbox.workflows.evaluate import (
    PROTOCOLS,
    evaluate,
    evaluate_fit_reports,
    evaluate_models,
    save_report,
)
from glassbox.workflows.holdout import evaluate_holdout


def _horizons(value: str) -> tuple[float, ...]:
    try:
        result = tuple(dict.fromkeys(float(item.strip()) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "horizons must be comma-separated numbers"
        ) from error
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("horizons must be positive")
    return result


def _named_path(value: str) -> tuple[str, Path]:
    """Parse ``NAME=PATH``; a bare path is named by its file stem."""

    name, separator, path = value.partition("=")
    if not separator:
        return Path(value).stem, Path(value)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("a named model must be given as NAME=PATH")
    return name.strip(), Path(path.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="*",
        type=Path,
        metavar="PATH",
        help=(
            "MODEL then the trajectory NPZ files to score it on; with "
            "--model, --hold-out or --fit-reports every path is a trajectory"
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=tuple(PROTOCOLS),
        default="windowed",
        help="scoring convention (default: windowed)",
    )
    parser.add_argument(
        "--model",
        action="append",
        type=_named_path,
        metavar="NAME=PATH",
        default=[],
        help="score this named model too; repeat to compare several models",
    )
    parser.add_argument(
        "--corpus",
        choices=tuple(REFERENCE_CORPORA),
        help="check the flights against this corpus's published evaluation split",
    )
    parser.add_argument("--report", type=Path, help="output report JSON")
    parser.add_argument(
        "--horizons",
        type=_horizons,
        help="comma-separated rollout horizons in seconds; the policy's by default",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        help="maximum prediction step for a per-step protocol such as nanodrone",
    )
    parser.add_argument(
        "--same-flight",
        action="store_true",
        help=("declare that these flights were not withheld from the fit"),
    )
    parser.add_argument("--json", action="store_true", help="print the whole report")

    holdout = parser.add_argument_group("leave-one-label-out runner")
    holdout.add_argument(
        "--hold-out",
        metavar="KEY",
        help="trajectory label to fold on, for example profile or source_group",
    )
    holdout.add_argument("--output-dir", type=Path, help="directory for the folds")
    holdout.add_argument("--training-horizons", type=_horizons, default=(0.1, 0.5, 2.0))
    holdout.add_argument(
        "--evaluation-horizons", type=_horizons, default=(0.1, 0.5, 1.0, 2.0)
    )
    holdout.add_argument("--steps", type=int, default=400)
    holdout.add_argument("--learning-rate", type=float, default=0.02)
    holdout.add_argument("--endpoint-weight", type=float, default=3.0)
    holdout.add_argument("--stability-regularization", type=float, default=0.01)
    holdout.add_argument("--run-no-lag-ablation", action="store_true")
    holdout.add_argument(
        "--model-class",
        choices=("structured", "structured_residual"),
        default="structured",
    )
    holdout.add_argument("--no-resume", dest="resume", action="store_false")
    holdout.add_argument(
        "--limit-folds",
        type=int,
        metavar="N",
        help=(
            "fit only the first N folds; a shortened run for exercising the "
            "chain, recorded as such in the summary's fold selection"
        ),
    )

    characterization = parser.add_argument_group("fit-report characterization")
    characterization.add_argument(
        "--fit-reports",
        nargs="+",
        type=_named_path,
        metavar="NAME=PATH",
        help="fit reports whose held-out horizon tables are scored",
    )
    characterization.add_argument(
        "--score-horizons",
        type=_horizons,
        help="horizons the score is taken over, when fewer than those measured",
    )
    return parser


def _run_holdout(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.output_dir is None:
        parser.error("--hold-out requires --output-dir")
    if not args.path:
        parser.error("--hold-out requires at least one trajectory")
    evaluate_holdout(
        args.path,
        hold_out=args.hold_out.replace("-", "_"),
        spec=FitSpec(
            horizons_s=args.training_horizons,
            evaluation_horizons_s=args.evaluation_horizons,
            steps=args.steps,
            learning_rate=args.learning_rate,
            model_class=args.model_class,
            ablations=("no_lag",) if args.run_no_lag_ablation else (),
            loss=LossPolicy(
                endpoint_weight=args.endpoint_weight,
                stability_regularization=args.stability_regularization,
            ),
        ),
        output_dir=args.output_dir,
        resume=args.resume,
        fold_limit=args.limit_folds,
    )


def _run_fit_reports(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.report is None:
        parser.error("--fit-reports requires --report")
    report = evaluate_fit_reports(
        dict(args.fit_reports),
        protocol=args.protocol,
        horizons_s=args.horizons,
        score_horizons_s=args.score_horizons,
    )
    for name, model in report["models"].items():
        metrics = model["aggregate_horizon_rollouts"]
        label = max(metrics, key=lambda item: float(item.rstrip("s")))
        print(
            f"{name}: score/{report['baseline']}={model['score_vs_baseline']:.3f}  "
            f"{label} position={metrics[label]['position_rmse_m']:.3f}m  "
            f"attitude={metrics[label]['attitude_rmse_deg']:.2f}deg"
        )
    print(f"selected={report['selected_model']}")
    _write(report, args)


def _corpus_block(name: str) -> dict[str, Any]:
    corpus = REFERENCE_CORPORA[name]
    return {
        "name": corpus.name,
        "doi_or_url": corpus.citation.doi_or_url,
        "license": corpus.citation.license,
        "pinned_version": corpus.citation.pinned_version,
        "evaluation_split": corpus.validation_split,
    }


def _load(args: argparse.Namespace, paths: Sequence[Path]) -> list[Any]:
    if args.corpus is not None:
        _, trajectories = REFERENCE_CORPORA[args.corpus].load_evaluation_trajectories(
            paths
        )
        return trajectories
    return [load_trajectory_npz(path) for path in paths]


def _write(report: dict[str, Any], args: argparse.Namespace) -> None:
    if args.report is not None:
        save_report(report, args.report)
        print(f"wrote {args.report}")
    elif args.json:
        print(json.dumps(report, indent=2))


def _print_horizons(model: dict[str, Any], name: str | None = None) -> None:
    prefix = "" if name is None else f"{name}: "
    selected = model.get("selected_horizons")
    if selected is not None:
        for step, metrics in selected.items():
            print(
                f"{prefix}h={step} ({metrics['time_s']:.3f}s): "
                f"position={metrics['position_mae_m']:.5f}m  "
                f"velocity={metrics['velocity_mae_m_s']:.5f}m/s  "
                f"attitude={metrics['attitude_mae_rad']:.5f}rad  "
                f"angular_velocity={metrics['angular_velocity_mae_rad_s']:.5f}rad/s"
            )
        return
    for label, metrics in model["horizon_rollouts"].items():
        print(
            f"{prefix}{label}: position={metrics['position_rmse_m']:.4f}m  "
            f"velocity={metrics['velocity_rmse_m_s']:.4f}m/s  "
            f"attitude={metrics['attitude_rmse_deg']:.3f}deg  "
            f"rate={metrics['angular_velocity_rmse_rad_s']:.4f}rad/s"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = [
        name
        for name, chosen in (
            ("--hold-out", args.hold_out is not None),
            ("--fit-reports", args.fit_reports is not None),
        )
        if chosen
    ]
    if len(modes) > 1:
        parser.error(f"{' and '.join(modes)} run different evaluations")
    if args.hold_out is not None:
        _run_holdout(args, parser)
        return
    if args.fit_reports is not None:
        _run_fit_reports(args, parser)
        return

    independent = not args.same_flight
    if args.model:
        trajectories = _load(args, args.path)
        report = evaluate_models(
            dict(args.model),
            trajectories,
            protocol=args.protocol,
            horizons_s=args.horizons,
            independent_holdout=independent,
            expected_spec=trajectories[0].spec.to_dict(),
        )
        score_key = f"score_vs_{report['baseline']}"
        for name, model in report["models"].items():
            _print_horizons(model["aggregate"], name)
            print(f"{name}: score/{report['baseline']}={model[score_key]:.3f}")
    else:
        if len(args.path) < 2:
            parser.error("a model and at least one trajectory are required")
        report = evaluate(
            args.path[0],
            _load(args, args.path[1:]),
            protocol=args.protocol,
            horizons_s=args.horizons,
            maximum_horizon_steps=args.max_horizon,
            independent_holdout=independent,
        )
        _print_horizons(report["model"])
        score = report.get("score_vs_baseline")
        if score is not None:
            print(f"score/{report['baseline']}={score:.3f}")
    if args.corpus is not None:
        report["corpus"] = _corpus_block(args.corpus)
    _write(report, args)


if __name__ == "__main__":
    main()
