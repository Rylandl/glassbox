"""Frozen, no-fit command swaps and physical-basis head identifiability audit."""

import argparse
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _online_trace_diagnostics import (
    HEAD_PARTS,
    _stage_fields,
    integrate_stages,
    physical_metrics,
)
from collect_throw import authenticate, binding, check_source, seal
from run_dart import ROOT, clean, write
from verify_baseline import arrays, digest, read, require

from glassbox import _dynamics as dynamics
from glassbox._learner_arrays import load_arrays

PROTOCOL = ROOT / "docs/harness/online-response-support-v1.json"


def physical_coordinates(norms):
    scale = np.r_[norms["body_scale"], norms["input_scale"], norms["input_scale"]]
    offset = np.zeros_like(scale)
    offset[6:9] = norms["body_mean"][6:9]
    return scale, offset


def physical_design(current, history, hidden, norms):
    """Invertibly express the supported head coordinates in physical units."""
    scale, offset = physical_coordinates(norms)
    physical = current * scale + offset
    delayed = ((history - current[:, None]) * scale).reshape(len(current), -1)
    affine = np.column_stack((np.ones(len(current)), physical, delayed, hidden))
    left, right = np.triu_indices(current.shape[1])
    quadratic = physical[:, left] * physical[:, right]
    return affine, quadratic


def physical_coefficients(params, norms):
    """Transform existing coefficients, including quadratic shifts, without fitting."""
    scale, offset = physical_coordinates(norms)
    current = len(scale)
    memory = params["memory"].shape[1]
    delayed = params["linear"].shape[0] - current - memory
    linear = params["linear"] / norms["feature_scale"][:, None] * norms["output_scale"]
    quadratic = (
        params["quadratic"] / norms["quadratic_scale"][:, None] * norms["output_scale"]
    )
    affine = np.zeros((len(linear) + 1, 6))
    affine[0] = params["bias"] * norms["output_scale"]
    affine[1:] = linear
    affine[1 : current + 1] /= scale[:, None]
    affine[current + 1 : current + delayed + 1] /= np.tile(scale, delayed // current)[
        :, None
    ]
    shift = -offset / scale
    affine[0] += shift @ linear[:current]
    left, right = np.triu_indices(current)
    affine[0] += (shift[left] * shift[right]) @ quadratic
    np.add.at(affine, 1 + left, (shift[right] / scale[left])[:, None] * quadratic)
    np.add.at(affine, 1 + right, (shift[left] / scale[right])[:, None] * quadratic)
    return affine, quadratic / (scale[left] * scale[right])[:, None]


def original_polynomial(params, norms, current, history, hidden):
    sampled = np.column_stack(
        (current, (history - current[:, None]).reshape(len(current), -1), hidden)
    )
    left, right = np.triu_indices(current.shape[1])
    quadratic = current[:, left] * current[:, right]
    return (
        sampled / norms["feature_scale"] @ params["linear"]
        + quadratic / norms["quadratic_scale"] @ params["quadratic"]
        + params["bias"]
    ) * norms["output_scale"]


def svd_design(matrix):
    """Declared machine-precision rank; preserve a true empty left complement."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if not min(matrix.shape):
        return dict(
            left=np.eye(matrix.shape[0]),
            singular=np.empty(0),
            row_basis=np.empty((0, matrix.shape[1])),
            rank=0,
            tolerance=0.0,
        )
    left, singular, right = np.linalg.svd(matrix, full_matrices=True)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    return dict(
        left=left,
        singular=singular,
        row_basis=right[:rank],
        rank=rank,
        tolerance=float(tolerance),
    )


def support_rows(query, decomposition):
    coordinates = query @ decomposition["row_basis"].T
    projected = coordinates @ decomposition["row_basis"]
    null = query - projected
    summaries = []
    for q, p, n, c in zip(query, projected, null, coordinates):
        energy = float(q @ q)
        summaries.append(
            dict(
                energy=energy,
                rowspace_energy_fraction=float(p @ p) / energy if energy else None,
                right_null_energy_fraction=float(n @ n) / energy if energy else None,
                weighted_leverage=float(
                    np.sum(
                        (c / decomposition["singular"][: decomposition["rank"]]) ** 2
                    )
                ),
            )
        )
    return summaries, projected, null


def analyze_design(
    affine,
    quadratic,
    query_affine,
    query_quadratic,
    weights,
    beta_affine,
    beta_quadratic,
):
    """SVD and coefficient attribution are algebra, not fitted alternative dynamics."""
    root_weight = np.sqrt(weights[:, None])
    a, q = root_weight * affine, root_weight * quadratic
    combined = np.column_stack((a, q))
    query = np.column_stack((query_affine, query_quadratic))
    beta = np.vstack((beta_affine, beta_quadratic))
    decompositions = {
        name: svd_design(value)
        for name, value in (("affine", a), ("quadratic", q), ("combined", combined))
    }
    summary, output = (
        {},
        dict(
            affine=affine,
            quadratic=quadratic,
            weights=weights,
            query_affine=query_affine,
            query_quadratic=query_quadratic,
            beta_affine=beta_affine,
            beta_quadratic=beta_quadratic,
        ),
    )
    for name, queries in (
        ("affine", query_affine),
        ("quadratic", query_quadratic),
        ("combined", query),
    ):
        decomposition = decompositions[name]
        supported, projected, null = support_rows(queries, decomposition)
        summary[name] = dict(
            rank=decomposition["rank"],
            singular_values=decomposition["singular"].tolist(),
            tolerance=decomposition["tolerance"],
            query_support=supported,
        )
        output[name + "_query_projection"] = projected
        output[name + "_query_null"] = null
    basis = decompositions["combined"]["row_basis"]
    beta_null = beta - basis.T @ (basis @ beta)
    output["combined_beta_null"] = beta_null
    output["query_null_acceleration"] = query @ beta_null
    output["retained_weighted_null_acceleration"] = combined @ beta_null
    polynomial = query @ beta
    supported = query @ (beta - beta_null)
    np.testing.assert_allclose(
        polynomial, supported + output["query_null_acceleration"], rtol=1e-9, atol=1e-9
    )
    summary["combined"]["retained_null_max_abs"] = float(
        np.max(np.abs(output["retained_weighted_null_acceleration"]), initial=0)
    )
    summary["combined"]["null_acceleration"] = output[
        "query_null_acceleration"
    ].tolist()
    da = decompositions["affine"]
    rank = da["rank"]
    mapping = (
        da["row_basis"].T @ ((da["left"][:, :rank].T @ q) / da["singular"][:rank, None])
        if rank
        else np.zeros((a.shape[1], q.shape[1]))
    )
    independent = da["left"][:, rank:].T @ q
    query_residual = query_quadratic - query_affine @ mapping
    conditional = svd_design(independent)
    supports, projected, null = support_rows(query_residual, conditional)
    quadratic_null = beta_quadratic - conditional["row_basis"].T @ (
        conditional["row_basis"] @ beta_quadratic
    )
    contribution = query_residual @ beta_quadratic
    unsupported = query_residual @ quadratic_null
    reconstructed = (
        query_affine @ (beta_affine + mapping @ beta_quadratic) + contribution
    )
    np.testing.assert_allclose(polynomial, reconstructed, rtol=1e-9, atol=1e-9)
    summary["conditional_quadratic"] = dict(
        independent_rows=len(independent),
        rank=conditional["rank"],
        singular_values=conditional["singular"].tolist(),
        tolerance=conditional["tolerance"],
        query_support=supports,
        acceleration=contribution.tolist(),
        unsupported_acceleration=unsupported.tolist(),
        retained_null_max_abs=float(
            np.max(np.abs(independent @ quadratic_null), initial=0)
        ),
        reconstruction_max_abs=float(
            np.max(np.abs(polynomial - reconstructed), initial=0)
        ),
    )
    output.update(
        conditional_mapping=mapping,
        conditional_design=independent,
        conditional_query=query_residual,
        conditional_query_projection=projected,
        conditional_query_null=null,
        conditional_beta_null=quadratic_null,
        conditional_acceleration=contribution,
        conditional_unsupported_acceleration=unsupported,
        query_polynomial_acceleration=polynomial,
    )
    return summary, output


def load_capture(path):
    metadata, saved = load_arrays(path / "session.npz")
    require(
        metadata["format"] == "glassbox-online-fit-v4", "capture session format differs"
    )
    model = dynamics.VehicleSequenceModel.from_arrays(
        metadata["model"],
        {k: v for k, v in saved.items() if k.startswith(("param_", "norm_"))},
    )
    return model, saved, arrays(path / "context.npz"), arrays(path / "diagnostics.npz")


def cache_design(model, saved):
    rows, role_weights, old = [], [], []
    roles = [
        role for role in ("bootstrap", "recent") if len(saved[role + "_past_states"])
    ]
    params, norms = jax.tree.map(jnp.asarray, (model.params, model.norms))
    for role in roles:
        past, commands, current_command = (
            jnp.asarray(saved[role + "_" + key])
            for key in ("past_states", "past_inputs", "future_inputs")
        )
        applied, history, hidden = dynamics._history(
            params, norms, past, commands, model.delay_steps, model.dt_s
        )
        current = dynamics.current_features(
            past[:, -1], current_command[:, 0], applied, norms
        )
        current, history, hidden = map(np.asarray, (current, history, hidden))
        rows.append(physical_design(current, history, hidden, model.norms))
        role_weights.append(np.full(len(current), 1 / (len(roles) * len(current))))
        old.append(
            original_polynomial(model.params, model.norms, current, history, hidden)
        )
    return (
        *(np.concatenate([row[k] for row in rows]) for k in range(2)),
        np.concatenate(role_weights),
        np.concatenate(old),
    )


def inspect_point(path):
    model, saved, _context, stages = load_capture(path)
    with jax.enable_x64(True):
        affine, quadratic, weights, old = cache_design(model, saved)
        current = stages["factor_1_features"][:, :2].reshape(
            -1, stages["factor_1_features"].shape[-1]
        )
        history = np.broadcast_to(
            stages["factor_1_history"],
            (len(current), *stages["factor_1_history"].shape),
        )
        hidden = np.broadcast_to(
            stages["factor_1_hidden"], (len(current), *stages["factor_1_hidden"].shape)
        )
        query_a, query_q = physical_design(current, history, hidden, model.norms)
        beta_a, beta_q = physical_coefficients(model.params, model.norms)
        np.testing.assert_allclose(
            affine @ beta_a + quadratic @ beta_q, old, rtol=1e-9, atol=1e-9
        )
        original = (
            stages["factor_1_head_contributions"][:, :2, :5].sum(axis=-2).reshape(-1, 6)
        )
        actual = query_a @ beta_a + query_q @ beta_q
        np.testing.assert_allclose(actual, original, rtol=1e-9, atol=1e-9)
        summary, output = analyze_design(
            affine, quadratic, query_a, query_q, weights, beta_a, beta_q
        )
    summary.update(
        model_fingerprint=model.fingerprint,
        query_labels=[
            "observed_origin_and_substep_0_start",
            "substep_0_midpoint",
            *[
                f"substep_{i}_{name}"
                for i in range(1, len(stages["factor_1_states"]))
                for name in ("start", "midpoint")
            ],
        ],
        retained_origins=len(affine),
        physical_reconstruction_max_abs=float(np.max(np.abs(actual - original))),
        query_neural_acceleration=stages["factor_1_head_contributions"][:, :2, 5]
        .reshape(-1, 6)
        .tolist(),
    )
    output["query_original_polynomial_acceleration"] = original
    output["query_neural_acceleration"] = stages["factor_1_head_contributions"][
        :, :2, 5
    ].reshape(-1, 6)
    return summary, output


def command_swaps(reference):
    captures = {
        row: load_capture(reference / "fixedwing-81" / str(row)) for row in (156, 157)
    }
    require(
        captures[156][0].fingerprint == captures[157][0].fingerprint,
        "swapped contexts do not share identical model weights",
    )
    summary, output = {}, {}
    with jax.enable_x64(True):
        for origin, (model, _saved, context, _stages) in captures.items():
            for command_row in (156, 157):
                command = captures[command_row][2]["command"]
                name = f"context_{origin}_command_{command_row}"
                entry = dict(
                    factual=origin == command_row,
                    model_fingerprint=model.fingerprint,
                    head_part_order=HEAD_PARTS,
                    context_origin=origin,
                    command_origin=command_row,
                    resolutions={},
                )
                for factor in (1, 64):
                    stage = integrate_stages(
                        model,
                        context["past_states"],
                        context["past_inputs"],
                        command,
                        factor,
                    )
                    output[f"{name}_factor_{factor}_prediction"] = stage["prediction"]
                    entry["resolutions"][str(factor)] = dict(
                        prediction=stage["prediction"].tolist()
                    )
                    if origin == command_row:
                        np.testing.assert_allclose(
                            stage["prediction"],
                            _stages[f"factor_{factor}_prediction"],
                            rtol=1e-10,
                            atol=1e-10,
                        )
                        entry["resolutions"][str(factor)]["factual_truth_errors"] = (
                            physical_metrics(stage["prediction"], context["truth"])
                        )
                    if factor == 1:
                        fields = jax.tree.map(
                            np.asarray,
                            _stage_fields(
                                model.params,
                                model.norms,
                                jnp.asarray(stage["states"][0, :1]),
                                jnp.asarray(stage["filtered"][0, :1]),
                                jnp.asarray(command),
                                jnp.asarray(stage["history"]),
                                jnp.asarray(stage["hidden"]),
                            ),
                        )
                        output[f"{name}_start_head_parts"] = fields[
                            "head_contributions"
                        ][0]
                        entry["start_head_parts"] = fields["head_contributions"][
                            0
                        ].tolist()
                summary[name] = entry
    return summary, output


def run(reference, output, protocol=PROTOCOL):
    frozen = read(protocol)
    authenticate(reference, frozen["reference"]["manifest_sha256"])
    bound = binding(protocol)
    reference_binding = read(reference / "binding.json")
    require(
        digest(ROOT / "src/glassbox/_dynamics.py")
        == reference_binding["source"]["files"]["src/glassbox/_dynamics.py"],
        "captured dynamics engine differs",
    )
    require(
        bound["runtime"] == read(reference / "binding.json")["runtime"],
        "pinned diagnostic runtime differs",
    )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "attempt.json", dict(status="started", optimizer_steps=0))
    write(output / "binding.json", bound)
    shutil.copy2(protocol, output / "protocol.json")
    shutil.copy2(reference / "manifest.json", output / "reference-manifest.json")
    summaries = {}
    selected = read(reference / "protocol.json")["cases"]
    for case in selected:
        for origin in case["captures"]:
            path = reference / case["id"] / str(origin)
            destination = output / case["id"] / str(origin)
            destination.mkdir(parents=True)
            for name in ("session.npz", "context.npz", "diagnostics.npz"):
                shutil.copy2(path / name, destination / name)
            summary, values = inspect_point(path)
            write(destination / "support.json", summary)
            np.savez_compressed(destination / "support.npz", **values)
            summaries[f"{case['id']}/{origin}"] = summary
            print(
                json.dumps(
                    dict(
                        point=f"{case['id']}/{origin}",
                        affine_rank=summary["affine"]["rank"],
                        conditional_rank=summary["conditional_quadratic"]["rank"],
                    )
                ),
                flush=True,
            )
    swaps, values = command_swaps(reference)
    write(output / "command-swaps.json", swaps)
    np.savez_compressed(output / "command-swaps.npz", **values)
    result = dict(
        protocol_id=frozen["id"],
        reference_authority=frozen["reference"]["manifest_sha256"],
        points=summaries,
        command_swaps=swaps,
        optimizer_steps=0,
        initialization_calls=0,
        observations_assimilated=0,
        read_only_calls=dict(
            captured_models=len(summaries),
            origin_feature_batches=2 * len(summaries),
            integration_rollouts=8,
            start_head_batches=4,
        ),
        complete=len(summaries) == 20,
    )
    write(output / "summary.json", result)
    check_source(bound)
    authority = seal(output, "glassbox-online-response-support-v1")
    print(
        json.dumps(
            clean(
                dict(
                    authority=authority, complete=result["complete"], optimizer_steps=0
                )
            )
        ),
        flush=True,
    )
    return authority


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
