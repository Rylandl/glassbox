"""Read-only backtracking of authenticated directions; no solver or learner changes."""

import argparse
import shutil
from collections import Counter
from functools import cache
from pathlib import Path

import audit_online as parent_audit
import numpy as np
from collect_throw import authenticate, binding, check_source, checkpoint, seal
from run_dart import ROOT, clean, write
from verify_baseline import arrays, digest, read, require

PROTOCOL = ROOT / "docs/harness/online-backtracking-v1.json"
ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625)
PARENT_AUTHORITY = "2287c318048a65f997db2a6a92f725e9a93ef8e0ee5cb5a932215f3bfb0bb7c0"


def _number(value):
    return float(value) if np.isfinite(value) else None


def _loss(raw, weight):
    squared = np.stack(
        [
            np.sum(raw[..., start:stop] ** 2, axis=-1)
            for start, stop in ((0, 3), (3, 6), (6, 15))
        ],
        axis=-1,
    )
    return float(
        np.sum(
            weight
            * np.where(
                squared <= 1, 0.5 * squared, np.sqrt(np.maximum(1.0, squared)) - 0.5
            )
        )
    )


def _comparison(value, reference):
    return "win" if value < reference else "loss" if value > reference else "equal"


def summarize(parent_summary, parent_arrays, trial_raw, *, conditioning_finite=True):
    """Report every alpha and the first pass; never choose the best observed loss."""
    a = parent_arrays
    theta, prior, gradient = (a[name] for name in ("theta", "prior", "gradient"))
    direction, projected = a["probe_delta"][3], a["probe_projected"][3]
    current_data = _loss(a["raw"], a["weight"])
    current_prior = float(0.5 * np.dot(theta, prior * theta))
    current = current_data + current_prior
    original = float(a["production_trial"])
    steps = []
    with np.errstate(all="ignore"):
        for alpha, raw in zip(ALPHAS, trial_raw):
            delta, jp = alpha * direction, alpha * projected
            candidate = theta + delta
            data = _loss(raw, a["weight"])
            penalty = float(0.5 * np.dot(candidate, prior * candidate))
            total = data + penalty
            predicted = float(
                -np.dot(gradient, delta)
                - 0.5
                * (
                    np.sum(a["weight"] * a["irls"] * jp**2)
                    + np.dot(delta, prior * delta)
                )
            )
            finite = bool(conditioning_finite and bool(a["finite"]))
            finite &= all(
                np.isfinite(value).all()
                for value in (
                    candidate,
                    delta,
                    jp,
                    raw,
                    gradient,
                    predicted,
                    current,
                    total,
                )
            )
            gain = (
                (current - total) / predicted if finite and predicted > 0 else -np.inf
            )
            steps.append(
                dict(
                    alpha=alpha,
                    finite=finite,
                    would_accept=bool(finite and total < current and gain >= 0.1),
                    data_loss=_number(data),
                    prior_loss=_number(penalty),
                    total_loss=_number(total),
                    data_improvement=_number(current_data - data),
                    prior_improvement=_number(current_prior - penalty),
                    improvement=_number(current - total),
                    predicted_reduction=_number(predicted),
                    gain=_number(gain),
                )
            )
    selected = next((i for i, step in enumerate(steps) if step["would_accept"]), None)
    kept = original if selected is None else steps[selected]["total_loss"]
    return dict(
        format="online-backtracking-point-v1",
        current=dict(
            data_loss=_number(current_data),
            prior_loss=_number(current_prior),
            total_loss=_number(current),
        ),
        original_four=dict(
            total_loss=original,
            instrumented_data_loss=parent_summary["probes"][1]["data_loss"],
            instrumented_prior_loss=parent_summary["probes"][1]["prior_loss"],
        ),
        parent_reference=parent_summary["reference"],
        steps=steps,
        selected_index=selected,
        selected_alpha=None if selected is None else ALPHAS[selected],
        selection_comparison="no_selection"
        if selected is None
        else _comparison(kept, original),
        retained_total_loss=kept,
        retained_comparison=_comparison(kept, original),
        hypothetical_residual_evaluations=5 if selected is None else selected + 1,
        work=dict(
            new_residual_calls=4,
            reused_residuals=1,
            cg_iterations=0,
            conditioning_calls=0,
            gradient_calls=0,
            initialization_calls=0,
            observe_calls=0,
            applied_updates=0,
        ),
    )


@cache
def _kernel(residual, delay, dt_s):
    import jax
    from jax.flatten_util import ravel_pytree

    @jax.jit
    def evaluate(params, norms, data, scale, direction, alphas):
        theta, unpack = ravel_pytree(params)
        return jax.lax.map(
            lambda alpha: residual(
                unpack(theta + alpha * direction), norms, data, scale, delay, dt_s
            ),
            alphas,
        )

    return evaluate


def evaluate_residuals(module, prepared, direction, alphas):
    """Only forward residual evaluations on the existing conditioned chart."""
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True):
        result = _kernel(module._residual, prepared["delay"], prepared["dt_s"])(
            prepared["params"],
            prepared["norms"],
            prepared["data"],
            prepared["scale"],
            jnp.asarray(direction),
            jnp.asarray(alphas, dtype=jnp.float64),
        )
        return np.asarray(result)


def check_protocol(protocol):
    require(protocol["id"] == "online-backtracking-v1", "unsupported protocol")
    require(protocol["ladder"] == list(ALPHAS), "backtracking ladder differs")
    require(
        protocol["parent"]["manifest_sha256"] == PARENT_AUTHORITY,
        "parent authority differs",
    )
    require(
        protocol["parent"]["protocol_id"] == "online-proposal-audit-v1",
        "parent protocol differs",
    )


def verify_parent(protocol):
    check_protocol(protocol)
    parent = Path(protocol["parent"]["path"]).resolve()
    result = parent_audit.verify(parent, PARENT_AUTHORITY)
    parent_protocol = read(parent / "protocol.json")
    require(
        parent_protocol["id"] == protocol["parent"]["protocol_id"],
        "parent protocol differs",
    )
    parent_audit.check_package_sources(parent_protocol, parent)
    require(
        result["snapshots"] == 46 and len(result["links"]) == 10,
        "parent coverage differs",
    )
    return parent, parent_protocol, result


def point_inputs(parent, protocol, version, case, origin):
    original = parent_audit.original_point(parent, protocol, version, case, origin)
    path = parent / version / case / str(origin)
    prepared = parent_audit.saved_prepared(path, original)
    summary, values = read(path / "diagnostics.json"), arrays(path / "diagnostics.npz")
    outcome = read(path / "reconstruction.json")
    identity = dict(
        version=version,
        case=case,
        origin=origin,
        original_after_model=outcome["core_after"],
        original_before_session=outcome["session_before"],
        parent_files={
            str(p.relative_to(parent)): digest(p)
            for p in (
                path / "prepared.npz",
                path / "diagnostics.json",
                path / "diagnostics.npz",
                path / "production.npz",
                path / "reconstruction.json",
            )
        },
    )
    return prepared, summary, values, identity


def points(protocol):
    for version in protocol["references"]:
        for case, origins in protocol["origins"].items():
            for origin in origins:
                yield version, case, origin


def aggregate(rows):
    def group(selected):
        selected_steps = [
            r["result"]["steps"][r["result"]["selected_index"]]
            for r in selected
            if r["result"]["selected_index"] is not None
        ]
        return dict(
            count=len(selected),
            selections=dict(
                Counter(str(r["result"]["selected_alpha"]) for r in selected)
            ),
            comparisons=dict(
                Counter(r["result"]["selection_comparison"] for r in selected)
            ),
            retained_comparisons=dict(
                Counter(r["result"]["retained_comparison"] for r in selected)
            ),
            reference_status=dict(
                Counter(r["result"]["parent_reference"]["status"] for r in selected)
            ),
            accepted_per_alpha=[
                sum(r["result"]["steps"][i]["would_accept"] for r in selected)
                for i in range(5)
            ],
            nonfinite_per_alpha=[
                sum(not r["result"]["steps"][i]["finite"] for r in selected)
                for i in range(5)
            ],
            selected_data_decreases=sum(
                s["data_improvement"] > 0 for s in selected_steps
            ),
            selected_prior_decreases=sum(
                s["prior_improvement"] > 0 for s in selected_steps
            ),
            hypothetical_residual_evaluations=sum(
                r["result"]["hypothetical_residual_evaluations"] for r in selected
            ),
        )

    return dict(
        all=group(rows),
        by_version={
            v: group([r for r in rows if r["version"] == v])
            for v in sorted({r["version"] for r in rows})
        },
        by_version_case={
            f"{v}/{c}": group([r for r in rows if r["version"] == v and r["case"] == c])
            for v, c in sorted({(r["version"], r["case"]) for r in rows})
        },
    )


def _close(actual, expected, name):
    require(np.shape(actual) == np.shape(expected), name + " shape differs")
    require(
        np.allclose(actual, expected, rtol=1e-8, atol=1e-10, equal_nan=True),
        name + " differs",
    )


def run(output, protocol=PROTOCOL, git="git"):
    import collect_throw
    from _online_backtracking_verification import verify_aggregate, verify_point

    output, protocol = output.resolve(), protocol.resolve()
    contract = read(protocol)
    check_protocol(contract)
    parent = Path(contract["parent"]["path"]).resolve()
    require(
        not output.is_relative_to(parent) and not parent.is_relative_to(output),
        "output overlaps parent",
    )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "attempt.json", dict(protocol=str(protocol)))
    try:
        collect_throw.GIT = git
        bound = binding(protocol)
        parent, parent_protocol, parent_verified = verify_parent(contract)
        require(
            bound["runtime"] == read(parent / "binding.json")["runtime"],
            "archived runtime differs",
        )
        shutil.copyfile(protocol, output / "protocol.json")
        shutil.copyfile(parent / "manifest.json", output / "parent-manifest.json")
        write(output / "binding.json", bound)
        write(output / "parent-verification.json", parent_verified)
        modules = {
            v: parent_audit.historical_module(parent, parent_protocol, v)
            for v in parent_protocol["references"]
        }
        rows = []
        for version, case, origin in points(parent_protocol):
            print(f"backtracking {version} {case} update {origin}", flush=True)
            prepared, parent_summary, values, identity = point_inputs(
                parent, parent_protocol, version, case, origin
            )
            shorter = evaluate_residuals(
                modules[version], prepared, values["probe_delta"][3], ALPHAS[1:]
            )
            trial = np.concatenate((values["probe_trial_raw"][3:4], shorter))
            result = summarize(
                parent_summary,
                values,
                trial,
                conditioning_finite=prepared["conditioning_finite"],
            )
            verify_point(
                result,
                parent_summary,
                values,
                trial,
                conditioning_finite=prepared["conditioning_finite"],
            )
            path = output / version / case / str(origin)
            path.mkdir(parents=True)
            checkpoint(path / "residuals.npz", trial_raw=trial)
            write(path / "identity.json", identity)
            write(path / "result.json", result)
            rows.append(dict(version=version, case=case, origin=origin, result=result))
        require(len(rows) == 46, "audit coverage differs")
        check_source(bound)
        verify_aggregate(aggregate(rows), rows)
        write(
            output / "summary.json",
            dict(
                complete=True,
                snapshots=len(rows),
                links=parent_verified["links"],
                aggregate=aggregate(rows),
                new_residual_calls=4 * len(rows),
                reused_residuals=len(rows),
                cg_iterations=0,
                conditioning_calls=0,
                gradient_calls=0,
                initialization_calls=0,
                observe_calls=0,
                applied_updates=0,
            ),
        )
    except BaseException as error:
        write(output / "failure.json", dict(complete=False, error=repr(error)))
        raise
    finally:
        authority = seal(output, "glassbox-online-backtracking-v1")
        print(dict(output=str(output), manifest_sha256=authority), flush=True)


def verify(output, authority, *, recompute=False, git="git"):
    from _online_backtracking_verification import verify_aggregate, verify_point

    output = output.resolve()
    authenticate(output, authority)
    require(not (output / "failure.json").exists(), "audit attempt failed")
    contract, bound = read(output / "protocol.json"), read(output / "binding.json")
    check_protocol(contract)
    require(
        digest(output / "protocol.json") == bound["protocol_sha256"],
        "protocol binding differs",
    )
    check_source(bound)
    require(
        digest(output / "parent-manifest.json") == PARENT_AUTHORITY,
        "copied parent authority differs",
    )
    parent, parent_protocol, parent_verified = verify_parent(contract)
    require(
        read(output / "parent-verification.json") == parent_verified,
        "parent verification differs",
    )
    require(
        bound["runtime"] == read(parent / "binding.json")["runtime"],
        "bound parent runtime differs",
    )
    modules = {}
    if recompute:
        import collect_throw

        collect_throw.GIT = git
        require(
            binding(PROTOCOL)["runtime"] == bound["runtime"],
            "recomputation runtime differs",
        )
        modules = {
            v: parent_audit.historical_module(parent, parent_protocol, v)
            for v in parent_protocol["references"]
        }
    rows, checks = [], []
    expected = {
        "attempt.json",
        "binding.json",
        "protocol.json",
        "parent-manifest.json",
        "parent-verification.json",
        "summary.json",
    }
    for version, case, origin in points(parent_protocol):
        prepared, parent_summary, values, identity = point_inputs(
            parent, parent_protocol, version, case, origin
        )
        path = output / version / case / str(origin)
        for name in ("identity.json", "result.json", "residuals.npz"):
            expected.add(str((path / name).relative_to(output)))
        require(read(path / "identity.json") == identity, "point identity differs")
        saved = arrays(path / "residuals.npz")
        require(set(saved) == {"trial_raw"}, "residual inventory differs")
        result, trial = read(path / "result.json"), saved["trial_raw"]
        checked = verify_point(
            result,
            parent_summary,
            values,
            trial,
            conditioning_finite=prepared["conditioning_finite"],
        )
        if recompute:
            fresh = evaluate_residuals(
                modules[version], prepared, values["probe_delta"][3], ALPHAS
            )
            _close(fresh, trial, "recomputed trial residuals")
        rows.append(dict(version=version, case=case, origin=origin, result=result))
        checks.append(dict(version=version, case=case, origin=origin, result=checked))
    require(
        set(read(output / "manifest.json")["files"]) == expected,
        "audit payload inventory differs",
    )
    saved_summary = read(output / "summary.json")
    aggregate_verified = verify_aggregate(saved_summary["aggregate"], rows)
    expected_summary = dict(
        complete=True,
        snapshots=len(rows),
        links=parent_verified["links"],
        aggregate=saved_summary["aggregate"],
        new_residual_calls=4 * len(rows),
        reused_residuals=len(rows),
        cg_iterations=0,
        conditioning_calls=0,
        gradient_calls=0,
        initialization_calls=0,
        observe_calls=0,
        applied_updates=0,
    )
    require(
        len(rows) == 46 and read(output / "summary.json") == expected_summary,
        "audit summary differs",
    )
    return dict(
        verified=True,
        mode="recompute" if recompute else "verify",
        snapshots=len(rows),
        links=parent_verified["links"],
        rows=checks,
        arithmetic_checks=sum(r["result"]["arithmetic_checks"] for r in checks),
        aggregate_verification=aggregate_verified,
        residual_calls=5 * len(rows) if recompute else 0,
        cg_iterations=0,
        conditioning_calls=0,
        gradient_calls=0,
        initialization_calls=0,
        observe_calls=0,
        applied_updates=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--protocol", type=Path, default=PROTOCOL)
    execute.add_argument("--git", default="git")
    for name in ("verify", "recompute"):
        command = sub.add_parser(name)
        command.add_argument("output", type=Path)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--git", default="git")
    args = parser.parse_args()
    if args.command == "run":
        run(args.output, args.protocol, args.git)
    else:
        print(
            clean(
                verify(
                    args.output,
                    args.manifest_sha256,
                    recompute=args.command == "recompute",
                    git=args.git,
                )
            )
        )


if __name__ == "__main__":
    main()
