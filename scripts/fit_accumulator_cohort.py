"""Run the six frozen fits with at most two workers; no timing comparison."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from collect_throw import seal
from qualify_accumulator import PROTOCOL
from run_dart import ROOT, write
from verify_baseline import digest, read, require


def run_arm(arm, output, baseline_root):
    root = baseline_root if arm == "baseline" else ROOT
    env = dict(
        os.environ, PYTHONPATH=str(root / "src") + os.pathsep + str(ROOT / "scripts")
    )
    results = {}
    for name in read(PROTOCOL)["models"]:
        folder = output / arm / name
        with (output / f"{arm}-{name}.log").open("x") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/qualify_accumulator.py"),
                    "fit",
                    "--arm",
                    arm,
                    "--model",
                    name,
                    "--output",
                    str(folder),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        manifest = folder / "manifest.json"
        results[name] = dict(
            returncode=result.returncode,
            manifest_sha256=digest(manifest) if manifest.exists() else None,
        )
        print(json.dumps(dict(arm=arm, model=name, **results[name])), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            arm: pool.submit(run_arm, arm, args.output, args.baseline_root)
            for arm in ("baseline", "candidate")
        }
        results = {arm: future.result() for arm, future in futures.items()}
    write(args.output / "results.json", results)
    print(
        json.dumps(
            dict(
                manifest_sha256=seal(
                    args.output, "glassbox-accumulator-offline-fits-v1"
                )
            )
        ),
        flush=True,
    )
    require(
        all(x["returncode"] == 0 for arm in results.values() for x in arm.values()),
        "one or more fits failed; complete logs retained",
    )


if __name__ == "__main__":
    main()
