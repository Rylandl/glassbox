"""No-fit posthoc angular feedback measurement from saved readout solves."""

import argparse
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from collect_throw import authenticate, binding, check_source, checkpoint, seal
from diagnose_readout_stability import jacobian
from run_dart import ROOT, write
from screen_cold_readout_curvature import readout_features
from screen_readout_sensitivity import numpy_jacobian
from verify_baseline import arrays, observed, read

from glassbox import _dynamics as core

PROTOCOL = ROOT / "docs/harness/readout-sensitivity-feedback-v1.json"


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def probe(params, norms, past, inputs, command, *, delay, dt_s):
    applied, history, hidden = core._history(
        params, norms, past[None], inputs[None], delay, dt_s
    )
    current = core.current_features(past[-1], command, applied[0], norms)
    phi = readout_features(params, norms, current, history[0], hidden[0])
    return phi, applied, history, hidden


@jax.jit
def autodiff(params, norms, state, command, applied, history, hidden):
    return jax.jacfwd(
        lambda rate: core._head(
            params, norms, state.at[3:6].set(rate), command, applied, history, hidden
        )[0][3:]
    )(state[3:6])


def summarize(data):
    result = {}
    for arm in ("curvature", "candidate"):
        eigen = np.linalg.eigvals(data[arm])
        result[arm] = dict(
            median_max_real=float(np.median(eigen.real.max(axis=-1))),
            maximum_real=float(eigen.real.max()),
            minimum_real=float(eigen.real.min()),
            positive_max_count=int(np.sum(eigen.real.max(axis=-1) > 0)),
            samples=int(np.prod(eigen.shape[:-1])),
        )
        radii = np.max(np.abs(np.linalg.eigvals(data[arm + "_full"])), axis=-1)
        result[arm]["full_recurrence_median_radius"] = float(np.median(radii))
        result[arm]["full_recurrence_maximum_radius"] = float(radii.max())
    result["maximum_autodiff_absolute_difference"] = float(
        data["autodiff_difference"].max()
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec, bound = read(PROTOCOL), binding(PROTOCOL)
    source = Path(spec["source"]["path"])
    authenticate(source, spec["source"]["manifest_sha256"])
    original = read(source / "protocol.json")["sources"]["tapes"]
    tapes = Path(original["path"])
    authenticate(tapes, original["manifest_sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", spec)
    write(args.output / "binding.json", bound)
    with jax.enable_x64(True):
        for name in spec["cases"]:
            info = read(source / name / "result.json")
            data = arrays(source / name / "predictions.npz")
            initial = arrays(source / name / "initial.npz")
            model = core.VehicleSequenceModel.from_arrays(
                info["initial_model_metadata"], initial
            )
            p, norm = jax.tree.map(jnp.asarray, (model.params, model.norms))
            mean0 = np.concatenate(
                (
                    model.params["linear"],
                    model.params["quadratic"],
                    model.params["bias"][None],
                    model.params["w2"],
                )
            )
            tape = arrays(tapes / "inputs" / (name + ".npz"))
            states, commands = observed(tape["states"]), tape["commands"].astype(float)
            saved = {
                "curvature": [],
                "candidate": [],
                "curvature_full": [],
                "candidate_full": [],
                "autodiff_difference": [],
            }
            for query, row in enumerate(data["conditional_rows"]):
                row = int(row)
                offset = row - info["first"]
                means = {
                    arm: mean0 if offset == 0 else data[arm + "_mean"][offset - 1]
                    for arm in ("curvature", "candidate")
                }
                matrices = {key: [] for arm in means for key in (arm, arm + "_full")}
                for j in range(5):
                    t, h = row + j, model.history_steps
                    phi, applied, history, hidden = probe(
                        p,
                        norm,
                        jnp.asarray(states[t - h : t + 1]),
                        jnp.asarray(commands[t - h : t]),
                        jnp.asarray(commands[t]),
                        delay=model.delay_steps,
                        dt_s=model.dt_s,
                    )
                    derivative = numpy_jacobian(
                        np.asarray(phi), initial, model.delay_steps
                    )[:, 3:6]
                    z = (states[t, 3:6] - model.norms["body_mean"][3:6]) / model.norms[
                        "body_scale"
                    ][3:6]
                    support = model.norms["motion_bound_scale"][3:6] / 4
                    tail = np.maximum(np.abs(z) - support, 0) / (3 * support)
                    slope = (1 - np.tanh(tail) ** 2) / model.norms["body_scale"][3:6]
                    for arm, mean in means.items():
                        jac = (
                            model.norms["output_scale"][3:6, None]
                            * (mean[:, 3:6].T @ derivative)
                            * slope[None, :]
                        )
                        matrices[arm].append(jac)
                        f, q = len(p["linear"]), len(p["quadratic"])
                        params = dict(
                            p,
                            linear=jnp.asarray(mean[:f]),
                            quadratic=jnp.asarray(mean[f : f + q]),
                            bias=jnp.asarray(mean[f + q]),
                            w2=jnp.asarray(mean[f + q + 1 :]),
                        )
                        matrices[arm + "_full"].append(
                            np.asarray(
                                jacobian(
                                    params,
                                    norm,
                                    (
                                        jnp.asarray(states[t : t + 1]),
                                        applied,
                                        history,
                                        hidden,
                                    ),
                                    jnp.asarray(commands[t]),
                                    dt_s=model.dt_s,
                                )
                            )
                        )
                        if query in (0, 7) and j == 0:
                            expected = np.asarray(
                                autodiff(
                                    params,
                                    norm,
                                    jnp.asarray(states[t]),
                                    jnp.asarray(commands[t]),
                                    applied[0],
                                    history[0],
                                    hidden[0],
                                )
                            )
                            np.testing.assert_allclose(
                                jac, expected, atol=1e-9, rtol=1e-10
                            )
                            saved["autodiff_difference"].append(
                                float(np.max(np.abs(jac - expected)))
                            )
                for key, value in matrices.items():
                    saved[key].append(value)
            saved = {k: np.asarray(v) for k, v in saved.items()}
            checkpoint(args.output / (name + ".npz"), **saved)
            result = summarize(saved)
            write(args.output / (name + ".json"), result)
            print(name, result, flush=True)
    check_source(bound)
    print(
        "manifest_sha256",
        seal(args.output, "glassbox-readout-sensitivity-feedback-v1"),
        flush=True,
    )


if __name__ == "__main__":
    main()
