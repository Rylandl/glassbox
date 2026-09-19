# Prospective public v4 numeric and AD fixtures

2026-09-19, design only. No proposed fixture, forecast, initializer, fitter, gradient
or numerical trial has been executed. Numbers below are prospective engineering
tolerances, not estimates fitted to new observations or physical adequacy limits.
The source ledger in `public-v4-numerical-port-design.md` applies.

## One inference policy and four error sources

Keep fitting/calibration float64 inside its owned context, persisted arrays NumPy
float64, and prediction ordinary ambient-JAX arithmetic with no internal flag
change. Both ambientFalse and ambientTrue eager and outer JIT/AD are mandatory.
Use a separate source-pinned research oracle in float64, not the new public core
as its own reference. Numerical fixture construction is test-only, not a consumer
model/precision option or a model-fitting/performance claim.

For original float64 inputs x and parameter/norm tree theta, define Q as cast to
float32 and back to float64 (including all nine parameter and eight norm arrays):

- Y0 = F64(theta, x): original-input reference.
- Y1 = F64(theta, Q(x)): same quantized caller values as default32 receives.
- Y2 = F64(Q(theta), Q(x)): additionally rounded parameter/normalization reference.
- Y3 = F32(theta, Q(x)): actual public default32 result, promoted only for scoring.

Report Y1-Y0 as input representation loss; Y2-Y1 as parameter/norm rounding;
Q(Y2)-Y2 as final output representation loss; and Y3-Q(Y2) as the remaining
arithmetic effect (including intermediate roundoff). These telescope to Y3-Y0.
**Nominal integration gates compare Y3 to Y1, not Y2 or Q(Y2).** Thus parameter,
normalization, output and arithmetic errors remain charged to the public default32
path. Original-input loss is separately visible. Fresh physical truth comparisons
still score Y3 against truth, without subtracting any error component.

The decomposition requires only prediction with fixed arrays; never refit or
reinitialize the quantized oracle. An altered diagnostic theta tree must not be
saved over the actual fitted artifact or presented as another learned revision.

## Fixed small nominal fixture roster

Three deterministic fixtures exercise arbitrary dimensions, different histories,
and coarse/non-dividing timing. All use width32/memory8, batch3, no latent input,
and exactly the production feature/filter/quadratic implementation.

| ID | d | m | dt | delay P | context C | maximum H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scalar-coarse | 1 | 1 | .25 | 1 | 2 | 1 |
| rectangular-input | 2 | 3 | .05 | 2 | 10 | 5 |
| rectangular-state | 5 | 2 | .02 | 5 | 25 | 12 |

The final horizon is 12 from the existing NumPy rint(12.5) rule, not 13. For each,
run H=1 and H=maximum (deduplicate scalar), single and batch3, and vmap(single).
State/input scales and offsets are bounded and fully specified:
`mu_x[i]=(-1)^i*2*(i+1)`, `sx[i]=2^((i mod 3)-1)`,
`mu_u[j]=(-1)^j*.2`, `su[j]=2^((j mod 3)-2)`.
Feature/interaction/autonomous scales are ones; delta scales are .1.

For b=0,1,2 and integer t, define normalized observation/input samples:
`zx=.15*sin(.37*t+.29*(i+1)+.11*b)+.05*cos(.13*t*(i+1)+.17*b)`;
`zu=.20*cos(.23*t+.31*(j+1)+.07*b)+.04*sin(.11*t*(j+1))`.
Physical observations are mu_x+sx*zx at t=-C,...,0; past inputs mu_u+su*zu at
t=-C,...,-1; future inputs at t=0,...,H-1. Build in NumPy float64, then retain
original and Q32 versions explicitly.

All predictor paths must be active. A deterministic parameter fixture can be
defined without RNG or fitting as follows, with row-major k=1,...,array.size,
f=(P+1)*(d+m)+8, q=d*m and a=d*(d+1)/2:

- linear: `.01*sin(k)/sqrt(f)`, then subtract `.04*I_d` from its first d rows and
  add `.08*sin(i+j+1)/m` at rows d+j, output i (i,j are zero-based).
- interaction: `.02*sin(k)/sqrt(q)`; autonomous: `.02*cos(k)/sqrt(a)`.
- bias: `.01*cos(k)`; w1: `.2*sin(k)/sqrt(f)`; b1: `.05*sin(k)`;
  w2: `.03*cos(k)/sqrt(32)`; memory: `.2*cos(k)/sqrt(f)`;
  memory_bias: `.02*sin(k)`.

These are explicitly artificial arithmetic fixtures, not learned-accuracy
evidence. Exercise them through the maintained mean wrapper and archive boundary
where valid; do not fabricate calibration provenance to call them fitted models.
Separately apply the same mandatory runtime surface checks to the actual two
public fit artifacts on the protocol's fixed saved query roster. This prevents
an artificial fixture from being the only integration evidence.

## Scores, transformations and fixed tolerances

Report and gate every element using `abs(actual-reference) <= atol +
rtol*abs(reference)`, with **no averaging that can hide one failed component**.
Forecasts are compared in normalized coordinates `(y-mu_x)/sx`, with these scoring
operations in NumPy float64. This removes irrelevant absolute offsets from a
relative-error denominator. Directional output derivatives divide by sx. Reverse
gradients with respect to observations/commands multiply by their corresponding
sx/su, so derivative comparisons have dimensionless units.

| Mandatory check | atol | rtol |
| --- | ---: | ---: |
| Public/oracle x64 prediction; x64 eager/JIT/batch/vmap parity | 1e-12 | 1e-10 |
| Default32 eager/JIT/batch/vmap parity | 2e-6 | 2e-5 |
| Default32 prediction versus Y1 | 1e-5 | 1e-4 |
| x64 JVP and reverse gradient parity against x64 oracle | 1e-11 | 1e-9 |
| Default32 JVP/reverse gradient versus quantized-input x64 oracle | 2e-6 | 2e-3 |
| JVP/VJP directional duality, x64 | 1e-11 | 1e-9 |
| JVP/VJP directional duality, default32 | 2e-6 | 2e-4 |
| Central finite difference versus AD, x64 | 2e-8 | 2e-5 |
| Central finite difference versus AD, default32 | 2e-5 | 1e-2 |

These tolerances allow ordinary roundoff in matmul/tanh/recursive scan; they do
not permit exact-float32 claims. The AD finite-difference bound is deliberately
looser than same-function AD parity because differencing adds cancellation and
truncation error. Low-magnitude derivatives below its absolute allowance are not
claimed relatively resolved. Report max error and worst index even when passing.

Use gradients for all three differentiable inputs, not just future commands.
For a single query take fixed output coefficients `c[k,i]=sin(1+k*d+i)/(H*d)`
and scalar objective `L=sum(c*(y-mu_x)/sx)`. Run reverse `grad(...,argnums=(0,1,2))`,
ordinary outer jit of prediction, jit of gradient and gradient through outer jit,
and JVP of prediction. Build normalized directions for each input tensor as
`v[r]=sin(r+1)/sqrt(sum(sin(r+1)^2))` over its flattened indices. One direction
changes only past observations, one only past commands, one only future commands;
physical directions multiply by sx/su. Compare full JVP outputs and full gradient
arrays, plus `<c,Jv/sx>` against `<grad_physical,v_physical>`.

Finite differences use the same fixed directions with h64=2^-16 and h32=2^-7,
central `(L(x+h*v)-L(x-h*v))/(2*h)`. Do not choose h from observed convergence,
drop a direction after failure or differentiate through input quantization as if
the cast's discontinuities were the learned dynamics. Report achieved float32
perturbations; any nominal direction collapsing entirely is a fixture failure.

Also require finite values, expected shapes/dtypes, and structural causality:
perturbing the last future command cannot change earlier forecasts, and its JVP
prefix must be exactly zero. Require a nonzero resolved command effect in each of
the three artificial nominal fixtures (maximum normalized first-command JVP >1e-4
using the first command's first-channel unit direction), so an all-zero or
stop-gradient implementation cannot pass through permissive near-zero tolerances.
This minimum-effect threshold applies only to those artificial fixtures. On the
fixed real-model query roster, report the derivative magnitude without imposing a
minimum: all declared parity, finite-difference, finiteness, shape, dtype and
causality checks still apply, including to genuinely small learned responses.

Same-runtime archive round trips require exact parameter/norm/cache bytes and
fingerprints and exact predictions for the same execution path. Do not demand
bitwise equality between eager and compiled arithmetic, or between precisions.
Fresh processes begin with x64 unset/False and True separately. Capture global
configuration before and after import, load, successful/failed fit/update and all
prediction/AD paths; test restoration from the owned fit scope and thread isolation.
No Glassbox runtime options or environment prerequisites are supplied by consumers.

## Fixed diagnostic representability stress (not nominal exemptions)

Use exactly two variants of rectangular-input; leave normalized parameters,
commands and model architecture unchanged:

1. Add 2^30 to every state offset and physical observation. The float32 spacing
   near 2^30 is 128, vastly larger than the nominal state variation.
2. Replace offsets with `(-1)^i*2^16`, scales with `sx*2^-10`, and rebuild physical
   observations from the same zx. Float32 spacing near positive 2^16 is .0078125,
   again larger than these variations. Report actual spacing for every sign/binade.

Report the four error terms, input unique-value collapse, spacing/sx at inputs and
outputs, finite/nonfinite status, actual perturbation size, and AD sensitivity
versus the finite observable response. Output quantization can make a finite
perturbation produce no resolvable output even when an analytic AD tangent exists;
that is a representation limitation, not automatically a wrong derivative rule.
Do not require nominal finite-difference tolerances in these two diagnostic cases.

Any ordinary transformation exception, corrupted archive, global state mutation,
wrong causality or nominal fixture parity failure **blocks API integration**.
Stress must still execute ordinary transforms or return an explicit documented
numeric-domain failure; it cannot justify a dtype/tracer compilation exception.
Stress numerical disagreement documents a measured limitation and must not be
reported as learned physical-model error. If the proposed public claim includes
accuracy at these scales, that claim must fail or be narrowed explicitly.

Only these two prospectively named artificial cases are diagnostic. No real
held-out query, failed flight condition, inconvenient dimension or small derivative
can be retroactively moved into this category. Fresh default32 forecasts/responses
and envelope coverage retain the full protocol query roster and decision rules;
a failure there remains a public deployment finding. Passing AD tests establishes
calculus of the learned mean, not fidelity of derivatives to the physical system.
