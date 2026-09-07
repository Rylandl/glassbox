# Repeated identification uncertainty pilot

This one predeclared pilot completed six independent production `DynamicsBelief.absorb`
updates, each on a separate 3 s synthetic training flight. Its combined rate
uncertainty exceeds observed squared errors in this design, but does **not** establish
that the recovery covariance counts calibrated sampling variance twice.

The target is the known recovery arm-ratio shift of 0.20; the common prior is the
unchanged deterministic recovery parameter grid with the library innovation noise
floor. There is no process or observation noise. Replication comes from independently
seeded training excitation, with one bounded update per fit, rather than a converged
batch optimizer. Evaluation starts from known resting physical and true-hover
actuator states. Training retains the production actuator history reconstruction;
its estimation effects are not separately identified.

The design was fixed before data generation: training seeds 91000–91005,
two calibration sources per fit (92000–92011),
and independent test sources 93000/93001. Calibration and test command sources are
1.2 s flights; the last 30 commands are replayed open-loop from matched initial
states, with prefixes scored at 0.1 and 0.6 s. Both test designs are shared across
fits, giving **six**, not twelve or 144, independent fitted estimates at each design.
Each E has two independently grouped endpoint samples per horizon. All 150 training
transitions were retained for each fit; each resolves 15 estimable directions.
No test or calibration evidence expands support or enters identification.

For each test design, the saved endpoint error vector is expressed in the truth
endpoint tangent chart. Its derivative at true parameters times fitted parameter
error is `l`; `r = error - l` is the nonlinear remainder. Saved full matrices obey
`error error' = ll' + rr' + lr' + rl'` (maximum absolute closure 8.82e-13).
The repeated-fit second moment is also split into squared sample mean and centered
sample variation, using denominator six. This is a descriptive finite-sample
identity, not an unbiased estimate of population squared bias. Predicted C uses
the production information model and nominal-chart sensitivity; different attitude
tangent origins preclude asserting full matrix equality to truth-chart moments.

At 0.6 s, averaging the two fixed test designs only for compact presentation:

| Coordinate | Actual second moment | E | Parameter term | Sum | Sample bias² | Sample centered variation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Roll rate | 5.188e-6 | 1.812e-5 | 5.641e-5 | 7.453e-5 | 3.383e-6 | 1.805e-6 |
| Pitch rate | 8.162e-6 | 8.321e-6 | 3.367e-5 | 4.199e-5 | 8.151e-6 | 1.086e-8 |

These are in (rad/s)². The roll-rate nonlinear remainder second moment is
approximately 4e-9 and its signed cross contribution 1.56e-7. Pitch is strongly
sample-bias dominated; its cross contribution is 1.38e-7. Thus a large predicted
parameter term cannot be equated with the observed centered sampling variation.
The deterministic plant means E includes parameter-induced error, but this alone
does not show that C measures the same component with the correct magnitude.
The exact scalar finite-population control separately demonstrates genuine overlap
when E and the parameter contribution are defined to represent the same variance:
actual 15, E 15, parameter 15, sum 30, zero remainder and cross term.

All 0.1 s evaluations are within each fit's declared support. Both 0.6 s test
rollouts for replicate 4 exceed support slightly (largest nominal utilization
1.01483); other replicates remain inside. The table includes those labeled cases
and is not an inside-support calibration claim. Full per-fit/per-test validity
arrays and counts are in the report. With two calibration sources, six fits,
fixed noise-floor assumptions, no observation noise, single-update bias and
float32 dynamics, this pilot supports no population coverage, covariance
subtraction, rescaling, or controller change. No timing claims are made.

Artifacts: [machine report](investigations/repeated-uncertainty-calibration/report.json),
[predeclared design](investigations/repeated-uncertainty-calibration/design.json), and
`arrays.npz` plus six saved beliefs in the same directory. Source hashes, import
path, all source seeds/hashes, prior and unresolved flags are included. Validity
was annotated from saved fits after completion without any additional fit.
The report preserves separate pilot, annotation and delivered-script hashes.
The delivered source adds annotation, formatting and explicit binding of the
immediately evaluated Jacobian closure's test inputs. The six fits were not
rerun after these changes. Exact historical source bytes were not archived,
so the final script is a reproduction recipe, not a byte-for-byte copy of the
executed pilot source; the original hashes remain intact.

Reproduce from this checkout (the initial pilot used baseline e15c088):

```sh
uv run python scripts/calibrate_repeated_uncertainty.py \
  --output /tmp/repeated-uncertainty-pilot
uv run pytest tests/test_repeated_uncertainty_calibration.py -q
```

Exactly one pilot was run, without tuning. The tests falsify omission of a negative
cross term, check the bias/variation identity, independent seed/hash separation,
full saved matrix decomposition, component addition and the support annotation.
