"""No-fit diagnosis of the failed65a6813 physical-worker precision transition.

Run each mode in a fresh process against the immutable binding4 source. This is
postfailure diagnostic evidence; it changes no qualification gate or model.
"""

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import jax
import numpy as np

from glassbox import LearnedDynamics
from glassbox.experimental.public_mean_implementation import verify
from glassbox.experimental.public_mean_physical_evaluation import calibration_evidence

p = argparse.ArgumentParser()
p.add_argument(
    "mode", choices=["direct32", "calibration_then32", "eager64_then32", "jit32_then64"]
)
p.add_argument("output")
a = p.parse_args()
base = Path("/private/tmp/glassbox-public-mean-qualification/artifacts/2026-09-19")
worker = Path("/private/tmp/glassbox-public-mean-physical-evaluation")
binding = (
    worker
    / "artifacts/2026-09-19/public-mean-qualification-v1-implementation-attempt4.json"
)
binding_sha = "f88698133bd211128284f04330953b370e18cc12bc3dfb1eea312a5ae6ce7364"
verify(binding, binding_sha)
assert not jax.config.x64_enabled
model_path = base / "public-mean-qualification-v1-flight-fit-cascade-attempt1/model.npz"
data = base / "public-mean-qualification-v1-physics-cascade-attempt1/data"
query = json.loads((data / "queries.json").read_text())[0]
assert query["kind"] == "factual" and query["history_eligible"]
query_path = data / query["path"]
model = LearnedDynamics.load(model_path)
with np.load(query_path, allow_pickle=False) as archive:
    values = tuple(archive[k] for k in ("past_states", "past_inputs", "future_inputs"))
result = dict(
    mode=a.mode,
    binding_sha256=binding_sha,
    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    query_sha256=hashlib.sha256(query_path.read_bytes()).hexdigest(),
    query=query,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    fits=0,
    predictions=[],
)
try:
    if a.mode == "calibration_then32":
        calibration_evidence(model)
    elif a.mode == "eager64_then32":
        with jax.enable_x64(True):
            np.asarray(model.predict(*values))
    methods = {k: jax.jit(model.predict) for k in ("32", "64")}
    modes = (False, True) if a.mode == "jit32_then64" else (False,)
    for use64 in modes:
        with jax.enable_x64(use64):
            y = np.asarray(methods["64" if use64 else "32"](*values))
            result["predictions"].append(
                dict(
                    x64=use64,
                    dtype=str(y.dtype),
                    finite=bool(np.isfinite(y).all()),
                    sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                )
            )
    result["status"] = "passed"
except Exception as e:
    result.update(
        status="failed",
        error_type=type(e).__name__,
        error=str(e),
        traceback=traceback.format_exc(),
    )
result["ambient_restored"] = not bool(jax.config.x64_enabled)
verify(binding, binding_sha)
with Path(a.output).open("x") as f:
    f.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(
    json.dumps(
        {k: result[k] for k in ("mode", "status", "predictions", "ambient_restored")},
        sort_keys=True,
    )
)
