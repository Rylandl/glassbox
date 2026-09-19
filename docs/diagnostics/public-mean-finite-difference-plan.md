# Fixed finite-difference diagnosis proposal

Prospective diagnostic only. The frozen public qualification remains failed on
11 Cascade float32 observation-history finite-difference checks. No source,
model, tolerance, query roster, qualification flag or finite-difference step is
changed retrospectively. Do not execute this proposal before a reviewed clean
commit and externally anchored request. This is not another fitting experiment.

## Roster and arithmetic

Use all 24 actual mean inputs from the already sealed numeric manifest, in its
original order, all three declared directions (past observations, past commands,
future commands), and exactly h={2^-7,2^-8,2^-9,2^-10}. Thus 288 cells. Keep every
cell; report the entire grid, with no chosen h or new acceptance threshold.

Reuse the exact saved float32 direction arrays and quantized caller values.
The original source trees remain immutable: public d12145e and isolated research
oracle 8b61830. Actual public evaluation remains default32, unbatched eager as in
the failed check. The independent oracle evaluates in float64 with all nine parameters,
eight norms, caller inputs and scalar-objective constants first rounded to float32 and
promoted to float64. This gives a common arithmetic function for isolating float32
operations. Do not refit, reinitialize or save the quantized tree as a revision.

For each direction and step save the actual float32 plus/minus input coordinates,
forecasts, objective scalars and final32 central quotient. At h=2^-7 require the
quotient to be byte-identical to the existing frozen saved fd_i before continuing.
The original normalized directions have L2=1 before physical sx/su scaling.

For float64 save both ideal endpoints Q(x)+/-h*Q(v), and the actual achieved32 endpoint
coordinates promoted to float64, including their forecasts and objective scalars. One
float64 objective JVP per direction supplies the rounded-model local derivative.
The float64 objective uses the exact rounded32 coefficients, state mean and scale,
so differences do not silently include another normalization policy.

## Read-only reductions

For each cell retain the failed gate's original32 JVP contraction and decompose
fd32 minus that contraction into a telescoping sum:

1. ideal-endpoint64 central difference minus rounded-model64 AD: finite-step
   approximation error on the fixed smooth oracle;
2. achieved-endpoint64 difference minus ideal-endpoint64 difference: input
   perturbation rounding;
3. recomputed64 scalar difference from saved32 forecasts minus achieved64
   difference: forecast arithmetic and output representation;
4. difference of actual float32 scalar objectives promoted to float64 minus the preceding
   difference: scalar objective arithmetic;
5. actual float32 quotient minus64 subtraction/division of those same32 scalars:
   final differencing arithmetic;
6. rounded-model64 AD minus the original gate's AD contraction: the explicit
   derivative/constant-quantization bridge, not silently called pure roundoff.

Save these terms with both endpoint scalars; ensure their sum reconstructs the
original discrepancy (arithmetic consistency tolerance `1e-12 * (1 + abs(discrepancy))`,
not a qualification tolerance). The full raw predictions allow independent
reduction without more forecasts. Compare error patterns across the fixed grid;
do not claim cancellation or truncation solely from a single h.

## Budget, provenance and execution

Two sequential fresh workers: public32 then oracle64. Exactly 576 explicit
public32 mean evaluations;576 ideal and576 achieved oracle64 mean evaluations;
72 oracle64 objective JVP evaluations. These are diagnostic queries on fixed
arrays, zero fits, updates, initializers, optimizer steps or simulator calls.
Each worker has 7,200 seconds timeout; no automatic retry. Preserve command,
environment, stdout/stderr, exit and any failed prefix in an exclusive output.
The driver validates all previous run payload hashes, complete source maps,
three clean committed roots (public/oracle/diagnostic), interpreter/runtime and
import locations. The supervisor rechecks original binding before and after.

Implementation: `src/glassbox/experimental/public_mean_fd_diagnosis.py`. Stdlib/NumPy
imports only at module scope; worker source imports occur after selecting its
single source root. Parent must copy the reviewed source into a committed
checkout and construct the request afterwards. Request exact fields:

- diagnostic_root, diagnostic_relative_path, diagnostic_commit,
  diagnostic_source_sha256: newly committed diagnostic source identity;
- binding and binding_sha256: original implementation-attempt3.json and
  9347bc7aeea30c40315feb551f89d683e14a227b42076feb3ca946ecee0fdd0a;
- manifest and manifest_sha256: actual-numeric-input-attempt1/manifest.json and
  5a8f1ebd6007537bb9f9d751cd138c7e0400e9ff7180c5cd2771ef31df132797;
- original and run_sha256: actual-numeric-attempt1 directory and
  ac86379c6794d5e0962c25b35590b09754f9b09077605b3737497c5c8ff4471e.

The externally supplied request SHA binds all fields. Launch with the pinned
interpreter, SCIPY_ARRAY_API=1, original public-root/src PYTHONPATH and usable
Git first on PATH. Children receive original public-root/src or oracle-root/src
PYTHONPATH and the declared false/true precision. The diagnostic's final run.json
binds outputs and its own committed source; it contains no replacement pass flag.

Only AST parsing and Ruff are permitted before the diagnostic commit; no imports
of the draft into a numerical runtime, forecasts or test evaluations have run.
