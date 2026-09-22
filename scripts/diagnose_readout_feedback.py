"""Saved-snapshot instantaneous angular feedback; no model fitting."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from collect_throw import authenticate, binding, check_source, checkpoint, seal
from diagnose_readout_stability import ablate, carry_from, model_from, with_mean
from run_dart import ROOT, write
from verify_baseline import arrays, observed, read

from glassbox import _dynamics as core

PROTOCOL = ROOT / "docs/harness/readout-stability-feedback-v1.json"
ARMS = ("baseline", "candidate", "no_lag_rate", "no_quadratic_rate")


@jax.jit
def rate_jacobian(params, norms, carry, command):
    def acceleration(rate):
        state = carry[0].at[0, 3:6].set(rate)
        return core._head(params, norms, state, command[None], *carry[1:])[0][0, 3:]

    return jax.jacfwd(acceleration)(carry[0][0, 3:6])


def summarize(data):
    result = {}
    for arm in ARMS:
        eigen = np.linalg.eigvals(data[arm])
        result[arm] = dict(
            max_real=eigen.real.max(axis=-1).tolist(),
            min_real=eigen.real.min(axis=-1).tolist(),
            median_max_real=float(np.median(eigen.real.max(axis=-1))),
            maximum_real=float(eigen.real.max()),
            minimum_real=float(eigen.real.min()),
            positive_max_real_count=int(np.sum(eigen.real.max(axis=-1) > 0)),
            samples=int(np.prod(eigen.shape[:-1])),
            negative_real_beyond_midpoint_count=int(
                np.sum((np.abs(eigen.imag) < 1e-9) & (eigen.real < -80))
            ),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec, bound = read(PROTOCOL), binding(PROTOCOL)
    previous = Path(spec["diagnostic_path"])
    authenticate(previous, spec["diagnostic_manifest_sha256"])
    sources = read(previous / "protocol.json")["sources"]
    for source in sources.values():
        authenticate(Path(source["path"]), source["manifest_sha256"])
    source, tapes = (Path(sources[k]["path"]) for k in ("full", "tapes"))
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", spec)
    write(args.output / "binding.json", bound)
    with jax.enable_x64(True):
        for name in spec["cases"]:
            info = read(source / name / "result.json")
            solved = arrays(source / name / "predictions.npz")
            initial = model_from(info, arrays(source / name / "initial.npz"))
            tape = arrays(tapes / "inputs" / (name + ".npz"))
            states, commands = observed(tape["states"]), tape["commands"].astype(float)
            result = {arm: [] for arm in ARMS}
            for row in solved["conditional_rows"]:
                row = int(row)
                n = row - info["first"]
                candidate = (
                    initial if n == 0 else with_mean(initial, solved["mean"][n - 1])
                )
                models = dict(
                    baseline=model_from(
                        info, arrays(previous / name / f"baseline-{row}.npz")
                    ),
                    candidate=candidate,
                    **{arm: ablate(candidate, arm) for arm in ARMS[2:]},
                )
                for arm, model in models.items():
                    p, norm = jax.tree.map(jnp.asarray, (model.params, model.norms))
                    query = []
                    for j in range(5):
                        t, h = row + j, model.history_steps
                        carry = carry_from(
                            p,
                            norm,
                            jnp.asarray(states[t - h : t + 1]),
                            jnp.asarray(commands[t - h : t]),
                            delay=model.delay_steps,
                            dt_s=model.dt_s,
                        )
                        query.append(
                            np.asarray(
                                rate_jacobian(p, norm, carry, jnp.asarray(commands[t]))
                            )
                        )
                    result[arm].append(query)
            data = {k: np.asarray(v) for k, v in result.items()}
            checkpoint(args.output / (name + ".npz"), **data)
            write(args.output / (name + ".json"), summarize(data))
            print(
                name,
                {k: v["median_max_real"] for k, v in summarize(data).items()},
                flush=True,
            )
    check_source(bound)
    print(
        "manifest_sha256", seal(args.output, "glassbox-readout-feedback-v1"), flush=True
    )


if __name__ == "__main__":
    main()
