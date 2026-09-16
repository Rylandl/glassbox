"""Post-hoc check: do two documented regressions also worsen the selection loss?"""

import argparse
import hashlib
import json

import numpy as np


def main(args):
    results = []
    for run, name, chosen in (
        ("known-01", "near_periodic-5101-fit0", 200),
        ("coverage-confirmation-paths-01", "near_periodic-7303-fit0", 600),
    ):
        directory = args.artifacts / run / name
        record = json.loads((directory / "result.json").read_text())
        scale = np.asarray(record["loss_scales"]["first_step_floor"])
        for regime in ("matched", "shifted"):
            path = directory / f"{regime}.npz"
            with np.load(path, allow_pickle=False) as arrays:
                errors = arrays["first_step_floor"] - arrays["targets"]
                loss = np.mean((errors / scale) ** 2, axis=(1, 2, 3))
            results.append(
                dict(
                    case=name,
                    regime=regime,
                    chosen_step=chosen,
                    comparator_step=1000,
                    chosen_evaluation_criterion_mse=float(loss[chosen // 100]),
                    final_evaluation_criterion_mse=float(loss[10]),
                    relative_increase=float(loss[chosen // 100] / loss[10] - 1),
                    source=str(path),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    args.output.write_text(
        json.dumps(
            dict(
                timing="Post-hoc diagnostic after observing coverage results.",
                meaning="Uses the unchanged first-step-floor selection loss on evaluation data. Does not choose or replace any model. The final checkpoint was already a frozen comparator in the path study.",
                results=results,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
