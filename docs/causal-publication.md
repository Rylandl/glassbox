# Causal actuator learner: matched prefixes and published revisions

The causal actuator formulation is the candidate public learner. It shares one
fitted nonlinear applied-command state between body force and rigid-body torque,
fits inertia and actuator momentum from the current episode, and uses no vehicle
class, layout metadata or pretraining. This report corrects the earlier
[full-state research comparison](causal-fullstate.md), which gave the candidate
extra earlier rows for inertia and actuator-state reconstruction on several quad
cases. Here every candidate fit and its command history begin at the frozen
benchmark's `prefix_begin_row`.

The same frozen eight recordings provide 263 conditional 250 ms forecast origins.
The first comparison refits every completed prefix; the second scores the
immutable model a background worker had actually published at each origin while
observations were replayed at their recorded cadence. In both, future issued
commands are conditioned on their recorded values and are not used for fitting.
The published-revision replay begins **after** fitting the benchmark's initial
prefix; it records initialization time separately. Its first-origin prediction
therefore does not establish that a cold-start controller had a model available
at that exact instant. Forecast computation is deferred until after the timed
replay so JAX compilation does not change publication timing. Predictor latency
and the complete Throw timeline still need separate qualification.

The errors below are 250 ms RMSE over every frozen origin. Each cell is
**refitted prefix / published revision / incumbent public learner**. Velocity
and body-rate errors use physical units. The publication run is one paced CPU
replay; scheduling can change which revision is available at an origin.

| Recording | Origins | Velocity, m/s | Body rate, rad/s |
| --- | ---: | ---: | ---: |
| fixedwing-80 | 14 | 0.356 / 0.355 / 0.506 | 0.299 / 0.303 / 0.345 |
| fixedwing-81 | 14 | 0.425 / 0.467 / 0.751 | 0.350 / 0.377 / 0.597 |
| quad-arm-115 | 54 | 0.032 / 0.141 / 0.257 | 0.077 / 0.222 / 0.841 |
| quad-arm-125 | 54 | 0.023 / 0.039 / 0.124 | 0.113 / 0.142 / 0.551 |
| quad-arm-135 | 3 | 0.369 / 0.802 / 0.596 | 0.529 / 0.828 / 6.237 |
| quad-change | 54 | 0.023 / 0.037 / 0.124 | 0.114 / 0.142 / 0.551 |
| paired-quad-fine | 35 | 0.0015 / 0.0072 / 0.0330 | 0.0058 / 0.0152 / 0.0027 |
| paired-quad-coarse | 35 | 0.0035 / 0.0084 / 0.0058 | 0.0109 / 0.0175 / 0.0037 |

The six frozen arm-125 counterfactual command-response probes average **0.0728**
relative body-rate Jacobian error for the candidate versus **0.3309** for the
incumbent. These probes use separate simulator interventions at recorded states;
only the recorded observations and issued commands enter the fit. Fixed-wing
counterfactual truth is absent. `quad-change` shares much of its early dynamics
with arm-125 and is not independent class evidence.

A post-initialization paced replay with an unpenalized body-force intercept had
one severe early fixed-wing-80 failure: a fit through row 21 used an offset of
about 474 m/s² and produced 6.28 m/s velocity error at row 31. A weak generic
ridge penalty on that offset (0.01, decaying relative to data) reduces it to
0.85 m/s² and the corresponding error to 0.41 m/s. The revised full matched
benchmark improves or nearly preserves every active case; its six-probe response
mean remains 0.0728. The published-revision fixed-wing-80 aggregate improves
from 1.711 to 0.355 m/s. This is a fitting prior shared by all systems, not a
vehicle-specific guard or selection rule.

Initialization fits took 0.12–0.38 s across the paced cases. Published models
averaged about 0.4 s behind observations on fixed-wing and 1.1–1.3 s on the
long active quad streams. Thus the fitted equation's accuracy and publication
cadence remain different claims. The short arm-135 stream has only three origins:
its published velocity loss is material, despite a much smaller angular error.
The paired flights are gentle; their angular losses are small in absolute units
but consistently favor the incumbent. No held-out vehicle class, measurement
noise, live Throw/Dart catch or calibrated uncertainty is established here.

A second paced run of the timing-sensitive fixedwing-80, quad-arm-125 and
quad-arm-135 cases gave respectively 0.348/0.301, 0.043/0.177 and 0.802/0.828
m/s / rad/s velocity/body-rate RMSE. These preserve the same engineering
conclusion while showing timing variation on the longer quad stream.

The matched-prefix evaluator was frozen in commits `4c90242` and `9146af0`;
the timed-publication evaluator in `15d93a5`. The generic force-offset change
and fixed-wing early regression are in `9c5f849`; the public lifecycle source
is in `6c30827`. Frozen source identities are in
[online-readout-benchmark.json](online-readout-benchmark.json). Saved predictions
and parameter snapshots are in local ignored directories
`artifacts/causal-matched-prefix-ridge/` and
`artifacts/causal-publication-ridge/`. Without refitting, verify them with:

```bash
PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 uv run python scripts/qualify_causal_public.py --suite full --output artifacts/causal-matched-prefix-ridge --verify
PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 uv run python scripts/qualify_causal_publication.py --suite full --output artifacts/causal-publication-ridge --verify
```

SHA-256 of the two `summary.json` files is respectively
`37e71e884dcf8952c4ef8dfd8a21274c65474511f65d1b62a784c83940d378a0`
and `dad2b3004156ad20774dcc2c82160402de28de233fb43c82d75d3a1f550ea149`.
