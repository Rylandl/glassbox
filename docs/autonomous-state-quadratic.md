# Autonomous state quadratic experiment

The frozen `autonomous-state-quadratic-v1` candidate is **not adopted**. Adding
learned state–state products improves the weighted factual/response scores by
15.4%/15.4% against the public learner, but makes the named Crazyflow angular
response failure substantially worse. The public recipe remains unchanged.

The [protocol](harness/autonomous-state-quadratic-v1.json) was committed at
`a607a37` before implementation, simulation or fitting; implementation and tests
were committed at `7137007`. The [result record](harness/autonomous-state-quadratic-v1-result.json)
anchors the evidence. This is one mechanism test, not an application qualification.

## Mechanism and matched comparison

The candidate adds the upper-triangular products of standardized current observed
signals, `x_i*x_j` for `i<=j`, alongside the prior state–command products. All
products enter only the direct output; memory and neural features are unchanged.
The joint ridge initialization uses training-only scales, identical penalties and
an unpenalized intercept. The added 120 features produce 1,800 extra coefficients
per simulator. No platform equations or branches enter the learner.

Public baseline, unchanged bilinear comparator and quadratic candidate all fit
the same 384 training and 256 development windows, with 1,000 Adam updates,
minibatch size 64, identical draws and development checkpoint selection. Both
older models reproduce their complete saved revisions exactly. Private module
instances reuse the frozen optimizer and wrapper without modifying historical
sources or globals. Matching updates does not imply equal FLOPs. Parameter
counts are 12,470 / 13,370 / 15,170 for Crazyflow and 3,945 / 4,620 / 6,420 for
Cascade (public / bilinear / quadratic).

Calibration recordings, physics, collection pilots, 42 condition cells per
simulator, observations, queries and validity rules are unchanged. The 168 test
parent seeds are original baseline seeds plus 2,000,000, disjoint from both
previously inspected cohorts. All arms and hold-current see the same queries.
Paired simulator branches remain evaluation-only; fitting uses ordinary
observations and issued commands.

## Fresh held-out physical errors

Primary-condition 250 ms endpoint component RMSE, **public / bilinear / quadratic**:

| Simulator | Quantity | Velocity, m/s | Body rate, rad/s | Rotation entries |
| --- | --- | --- | --- | --- |
| Crazyflow | Forecast | 0.16834 / 0.15680 / 0.09002 | 0.25001 / 0.25372 / 0.43706 | 0.10489 / 0.10785 / 0.04584 |
| Crazyflow | Command response | 0.18166 / 0.13933 / 0.09756 | 0.33770 / 0.60739 / 1.10416 | 0.12095 / 0.10700 / 0.08675 |
| Cascade | Forecast | 0.16570 / 0.15493 / 0.12574 | 0.11426 / 0.11502 / 0.09423 | 0.01828 / 0.01829 / 0.01084 |
| Cascade | Command response | 0.12533 / 0.08537 / 0.11417 | 0.06573 / 0.05679 / 0.05532 | 0.01240 / 0.01137 / 0.00822 |

The exact values are in the saved comparisons; the table is rounded. Ratios use
fixed physical floors and equal simulator weights, with primary scope weight
0.5 and four shifted scopes weighted 0.125 each. Signal groups and 50/150/250 ms
horizons receive equal weights within a scope. These are weighted geometric
means of RMSE ratios, not pooled physical MSEs or a percentage of accurate models.

| Reference | Factual ratio [paired 95% interval] | Response ratio [paired 95% interval] |
| --- | --- | --- |
| Public learner | 0.84566 [0.82051, 0.86916] | 0.84649 [0.83071, 0.86049] |
| Bilinear comparator | 0.88313 [0.85646, 0.91056] | 0.99742 [0.97814, 1.01555] |

Both comparisons use the same 1,000 parent-bootstrap draws within condition
cells. Intervals describe this conditional population; they are not evidence
across arbitrary systems. Relative to bilinear, aggregate response error is
essentially unchanged, despite stronger velocity and rotation results in some
parts of the benchmark. Cascade's 250 ms velocity response also regresses versus
bilinear, while remaining better than public.

Three frozen checks fail:

- The targeted Crazyflow rate response is 1.81788 times bilinear; acceptance
  required at most 0.80. It is 3.26963 times the public baseline.
- Crazyflow primary response aggregate ratio is 1.16521, above the 1.15 bound.
- Crazyflow primary 250 ms parent-RMSE p95 ratios are 1.79625 for factual body
  rates and 2.98188 for rate responses, above the 1.5 bound. In physical units
  these change from 0.48390 to 0.86920 rad/s and 0.43869 to 1.30813 rad/s.

All eligible predictions are finite. Other aggregate/scope/tail checks pass.
No thresholds were changed after seeing this result.

## Fit, population and evidence limits

Crazyflow selects quadratic step 700 with development normalized rollout loss
0.021586, versus 0.046501 public and 0.046706 bilinear, both at step zero.
Quadratic's own step-zero loss is already 0.022242: the saved evidence does not
isolate how much angular-response damage comes from initialization versus later
training. Cascade selects step 1,000 in all arms, with losses 0.171138 / 0.168339 /
0.044545. Better development forecast loss does not establish command-response
fidelity; no response supervision or response-based selection was used.

Crazyflow completes 129/180 parent collections; 51 cross the altitude validity
boundary, including 27 calibration and 24 test parents. Primary 250 ms truth is
available for 420/480 factual and 703/768 response queries. Scores are conditional
on these prefixes, and every missing slot remains visible. Cascade completes
180/180 parents, with all 480 factual and 576 response primary queries available.
The fresh population differs from previous experiments, so all improvement
claims use matched arms on this new cohort, not differences between old tables.

Candidate primary 250 ms component envelope coverage is 89.0–89.3% in Crazyflow
and 89.7–91.9% in Cascade across the three signal groups. This descriptive result
is not an envelope qualification: the same development windows select the
checkpoint and calibrate the envelope, and shifted populations remain separate.
No derivative, update, public API adoption or downstream controller qualification
follows from these residuals.

## Diagnosis and next experiment

Saved-array analysis shows under-response across all 24 Crazyflow primary cells,
all four commands and both probe signs. At 250 ms, roll/pitch truth-response RMS
is 1.843/1.852 rad/s; quadratic predictions are 0.379/0.761, versus bilinear
0.996/1.378. The additional rate-response MSE is 60.5% roll, 39.4% pitch and
0.08% yaw. The error already increases at the first native sample, so this is
not exclusively long-horizon degradation.

The selected development objective reproduces exactly from saved channel errors:
bilinear velocity/rate/rotation contributions are 0.001387/0.004689/0.040629;
quadratic contributions are 0.000659/0.012596/0.008331. Thus a large rotation gain
outweighs the rate loss in the existing normalized factual objective. This is
observed arithmetic, not proof that changing the objective will repair responses.

The next bounded hypothesis is independent command excitation in ordinary
training recordings, keeping the quadratic architecture and original development
recordings fixed. Collection-input correlation and command saturation are
exploratory diagnostics, not causal identification. A separate frozen protocol
must declare the command perturbations, fitting budgets, fresh unchanged-pilot
test cohort, and acceptance criteria before execution. The learner will still
receive only observed signals and actual issued commands.

## Verification and reproduction

The externally anchored bundle is
`artifacts/2026-09-18/autonomous-state-quadratic-v1/run.json`, SHA256
`8792cbb51b2d86a8d9e6aab68c1e394a5a968ec51aa28e3ccae349d9d0f24d55`,
with 14,041 sealed payloads. Fresh execution reproduces all 55,644 physical/data/
query arrays, four-arm saved predictions on 4,032 queries, 342,720 metric rows,
and both decisions/bootstrap distributions exactly. Replay does not refit the
optimizers. Both comparator refits did independently reproduce earlier revisions.

An independent NumPy audit checks 250 ms physical errors for all four arms,
120 pairwise physical comparisons, 240 tail records, the 12 public tail gates,
the targeted angular gate, complete query/metric rosters, cached fit arrays and
training-only product scales. Its other-horizon aggregate checks use reported
ratios, and its bootstrap check verifies shared draw hashes; it does not claim
independent regeneration of those scores or draws. Production replay covers
them. Twelve disposable-copy tamper cases reject raw and coherent changes to
models (including the new quadratic coefficients), histories, commands, source
claims, predictions plus metrics, decisions and physical trajectories.

All 125 focused tests pass and Ruff is clean. The implementation audit verifies
99 original sources, four simulator harness sources, six inherited interaction
files, six new files, unchanged optimizer/wrapper code and shared initialization.
Independent product-scale reductions agree within 4.45e-15. An initial audit-only
missing metadata-key error and its corrected rerun are retained; no fit,
implementation, protocol or evaluation result changed.

Use the pinned Python 3.12.12 environment with JAX 0.11.1, NumPy 2.5.3 and x64:

```sh
SCIPY_ARRAY_API=1 JAX_ENABLE_X64=1 \
PYTHONPATH=src:/private/tmp/glassbox-cascade-e8f6ba6/src \
/Users/ryland/autonomy/dart/.venv/bin/python -m glassbox.experimental.state_quadratic_experiment \
  replay crazyflow --output artifacts/2026-09-18/autonomous-state-quadratic-v1 \
  --expected-bundle-sha 8792cbb51b2d86a8d9e6aab68c1e394a5a968ec51aa28e3ccae349d9d0f24d55
```

Repeat with `cascade`. Reproduction requires the frozen simulator sources and
both prior bundles at the paths declared in the protocol. Stage logs and
independent/tamper/implementation audits are separately anchored by the result
record. The performance figure presents all three learned arms and both query
kinds; it does not hide the angular regression behind an aggregate improvement.
