# First held-out Crazyflow arm configurations

The [frozen protocol](harness/heldout-quad-v1.json) requested two Crazyflow arm
lengths absent from the model-development recordings: 0.85 and 1.40 times the
base geometry. The pinned pass6 controller generated each flight independently
of the candidate. The public `OnlineFit` started fresh from each flight's own
75 observed intervals (0.5 s context plus 0.25 s fitting data) and then
predicted before assimilating each later observation. It received motion,
issued commands and timing, not arm length, mixer, inertia, applied rotor
telemetry or simulator parameters. This tests factual conditional predictions
under a behavior policy; the model did not fly the vehicle.

| New arm ratio | Flight | 250 ms origins | Native one-step rate, rad/s | 250 ms rate, rad/s | Hold-rate reference, rad/s | 250 ms velocity, m/s | Hold-velocity reference, m/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.85 | Floor contact at 1.9 s | 2 | 0.040 | **19.827** | 9.073 | 1.653 | 3.352 |
| 1.40 | Full 10 s | 35 | 0.0058 | **0.455** | 1.165 | 0.436 | 0.971 |

The hold reference keeps measured world velocity and body rate at each origin;
it has no learned dynamics. Both cases improve 250 ms velocity, and the 1.40
case improves rate. The 0.85 case is a substantial angular recurrence failure:
its individual 250 ms rate endpoint errors are 6.135 and 27.360 rad/s. At the
second origin the rate error grows from 1.246 at 50 ms to 5.689 at 100 ms,
13.463 at 150 ms and 27.360 at 250 ms. The two origins begin at body-rate
norms 9.10 and 8.20 rad/s, so the evidence is about high-spin recovery with
very limited sample count. Its short flight is a failed controller outcome of
the *pinned behavior policy*, not a failed Glassbox-controlled catch.

The 0.85 streaming one-step rate error is only 0.040 rad/s over the 25 scored
transitions. This does not validate a frozen-origin 250 ms recursion: the
online fit changes after each observation, whereas a conditional forecast
holds its identified coefficients fixed. A diagnostic recomputation of the
rate head at the second origin found that its error on later *measured*
angular-rate increments also grows, reaching more than 1 rad/s per 10 ms
interval. This points to angular-model mismatch along the future command and
state path, not merely numerical instability after a simulated state leaves
the recorded path. The present rate head is affine in issued and delayed
commands, with per-axis damping/memory but no cross-axis rate coupling or
state-dependent command effectiveness. High-spin inertial coupling and
unidentified command response are hypotheses to test, not established causes.

For the first new shape in the final run, initialization took 1.919 s, the
first 250 ms forecast 0.381 s and the first update 0.422 s; warmed updates
were about 1.8 ms. The second flight reused JAX compilations in the same
process, so its short startup timings are not independent cold-start evidence.
No live new-model controller, counterfactual command-response truth, fixed-wing
held-out configuration or new vehicle class was tested.

The sealed [source](../artifacts/heldout-quad-v1/source/manifest.json) and
[evaluation](../artifacts/heldout-quad-v1/evaluation/manifest.json) packs retain
both flights, all forecast origins, one-step predictions, timings and per-origin
errors. The harness was committed at `51cf2a6` before either flight was
collected. The final evaluator replay from the copied source pack produced
forecasts and one-step arrays exactly equal to the initial run (maximum
absolute difference 0.0). Verify physical errors without fitting:

```bash
PYTHONPATH=src:scripts python scripts/qualify_heldout_quad.py verify \
  --source artifacts/heldout-quad-v1/source \
  --source-manifest-sha256 5a60009ea431da8609186cb4cf3bd45419991724215ceeabf385b2cd9084ea80 \
  --output artifacts/heldout-quad-v1/evaluation \
  --manifest-sha256 e516c24e78544cd3d758cb7b448aae432f5ebfe920b0c4eb3c9fd64d279cb713
```

The next model iteration should target **high-spin angular generalization** in
the single generic rate formulation. A useful structural screen is to test
physically constrained cross-axis inertial coupling and a more identifiable
command-response lag, with the existing known recordings as development data.
Because the 0.85 flight has now informed the design, a future improvement must
be judged on fresh configurations rather than reusing it as a blind holdout.
