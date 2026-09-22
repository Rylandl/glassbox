"""One-time metadata finalization and saved-array audit; no learning calls.

Run from the adopted checkout with PYTHONPATH=src:scripts. Experimental fit
archives remain sealed. Only recipe metadata changes in the new revisions.
"""

import copy
from pathlib import Path

import jax
import numpy as np
from collect_throw import authenticate, seal
from qualify_accumulator import dart_queries, predict_queries
from run_dart import write
from screen_accumulator import aggregate, summarize
from verify_baseline import arrays, digest, exact, read, require

from glassbox import LearnedDynamics
from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.learner import RECIPE

ROOT = Path("/Users/ryland/autonomy/glassbox/artifacts/nonlinear-temporal-v1")


def main():
    index = read(ROOT / "replay-index.json")
    fits = Path(index["fits"]["path"])
    authenticate(fits, index["fits"]["manifest_sha256"])
    evaluation = ROOT / "offline-evaluation/candidate"
    authenticate(evaluation, index["evaluation"]["authorities"]["candidate"])
    protocol = read(evaluation / "protocol.json")
    baseline = Path(protocol["baseline"]["path"])
    authenticate(baseline, protocol["baseline"]["manifest_sha256"])
    output = ROOT / "revisions"
    output.mkdir(exist_ok=False)
    models, identities = {}, {}
    for name in protocol["models"]:
        source = fits / "candidate" / name / "model.npz"
        meta, values = load_arrays(source)
        require(
            meta["recipe"]["id"] == "shared-vehicle-accumulator-v1", "unexpected recipe"
        )
        require(
            meta["report"]["optimization"]["recipe"] == RECIPE["id"],
            "core recipe differs",
        )
        updated = copy.deepcopy(meta)
        updated["recipe"] = copy.deepcopy(RECIPE)
        updated["report"]["recipe"] = copy.deepcopy(RECIPE)
        target = output / (name + ".npz")
        save_arrays(target, updated, values)
        new_meta, new_values = load_arrays(target)
        require(
            new_meta == updated and set(new_values) == set(values), "repack identity"
        )
        for key in values:
            exact(new_values[key], values[key], "unchanged array " + key)
        models[name] = LearnedDynamics.load(target)
        identities[name] = dict(
            path=str(target),
            sha256=digest(target),
            fingerprint=models[name].fingerprint(),
            source_path=str(source),
            source_sha256=digest(source),
            unchanged_array_count=len(values),
        )
    count = 0
    for cohort, contract in read(baseline / "flights/spec.json")["cohorts"].items():
        actual = predict_queries(
            models[contract["simulator"]],
            arrays(baseline / "flights" / (cohort + ".npz")),
        )
        expected = arrays(evaluation / (cohort + ".npz"))
        for key in actual:
            exact(actual[key], expected[key], cohort + "/" + key)
            count += 1
    data, _ = dart_queries(baseline)
    actual = np.asarray(
        jax.jit(models["dart"].predict)(
            data["past_states"], data["past_inputs"], data["future_inputs"]
        )
    )
    exact(
        actual, arrays(evaluation / "dart-heldout.npz")["prediction"], "Dart forecasts"
    )
    write(
        output / "result.json",
        dict(
            models=identities,
            recipe=RECIPE,
            arrays_unchanged=True,
            replayed_arrays=count + 1,
            fits=0,
            optimizer_calls=0,
            exact=True,
            change="Correct public recipe metadata left at accumulator-v1 during the experiment; no numerical or calibration change.",
        ),
    )
    revision_authority = seal(output, "glassbox-temporal-finalized-revisions-v1")

    # Supplemental, post-hoc comparison: do not replace frozen candidate results.
    online_spec = read(ROOT / "online/candidate/protocol.json")
    authenticate(ROOT / "online", protocol["online_evidence"]["manifest_sha256"])
    for key in ("parent", "initial_sessions"):
        authenticate(
            Path(online_spec[key]["path"]), online_spec[key]["manifest_sha256"]
        )
    comparisons = {}
    for arm in ("baseline", "candidate"):
        for reference, location, prediction_key in (
            (
                "pre_projection_accumulator",
                Path(online_spec["initial_sessions"]["path"]) / "candidate",
                "prediction",
            ),
            ("v8", Path(online_spec["parent"]["path"]), "candidate"),
        ):
            cases = []
            for name in online_spec["cases"]:
                folder = ROOT / "online" / arm / name
                data, info = (
                    arrays(folder / "predictions.npz"),
                    read(folder / "case.json"),
                )
                previous = arrays(location / name / "predictions.npz")
                exact(data["row"], previous["row"], "same online rows")
                exact(data["truth"], previous["truth"], "same online truth")
                info["reference_parameters"] = info["parameters"]
                cases.append(summarize(data, info, previous[prediction_key]))
            result = aggregate(cases)
            # Old screen thresholds and cost ratios do not govern this secondary comparison.
            for key in ("gates", "screen_passed", "adopted", "claim"):
                result.pop(key)
            for case in result["cases"]:
                for key in ("parameters", "parameter_ratio", "predict", "update"):
                    case.pop(key)
            comparisons[arm + "_vs_" + reference] = result
    write(
        ROOT / "prior-comparison.json",
        dict(
            posthoc=True,
            fits=0,
            model_calls=0,
            timing_comparison=False,
            comparisons=comparisons,
        ),
    )
    print(dict(revisions_manifest_sha256=revision_authority, replayed_arrays=count + 1))


if __name__ == "__main__":
    main()
