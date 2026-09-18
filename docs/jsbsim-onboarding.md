# JSBSim onboarding audit

The first JSBSim iteration qualifies the recording and replay machinery before
fitting Glassbox. It is a setup audit, not evidence of learned accuracy or flight
capability. The [frozen protocol](harness/jsbsim-onboarding-v1.json) and
[release inventory](harness/jsbsim-onboarding-v1-inventory.json) define every case.

The JSBSim 1.3.1 source release contains 64 modern configurations, two legacy or
template XML candidates, and 61 aircraft directories. The directory inventory
includes LM, which has documentation but no configuration, while the root-level
aircraft template is an additional candidate outside those directories. All
candidates remain visible, including unsupported, passive and missing-setup cases.

The adapter selects a trim-free `reset00.xml` when available, otherwise the first
trim-free initialization file in lexical order. Thirty-nine candidates have a
selected initialization under this rule. It loads every candidate, including
those with no initialization. It neither invents a universal flight condition nor
repairs models after observing results. Shipped ground starts, engine-off starts
and any engine-running initialization remain exactly that; short finite motion
does not establish a useful flight regime.

Each initialized parent attempts two seconds at 20 Hz, using six JSBSim integration
steps per recorded interval. Commands comprise aileron, elevator, rudder and each
engine's throttle. They describe writes to the simulator's command interface;
they are not measured physical actuator positions. Values are reasserted at each
native step and readbacks record any internal overwrites. Observations include NED
velocity, body rates in forward-right-down coordinates, body-to-NED orientation,
and altitude above mean sea level, in declared SI units. Raw simulator observations
and timestamps allow a separate validator to check conversion and timing.

A fresh instance replays the complete parent. Each signed command probe also starts
fresh and replays the first second before changing one command for 250 ms. Probe
activity is a measured finite-amplitude response under that initialization and
horizon. An inactive probe establishes neither uncontrollability nor successful
learning. Missing/failed branches remain unavailable. No training labels enter a
learner in this iteration.

## Running and checking the audit

Use an isolated environment with the project's dependencies and `jsbsim==1.3.1`.
Download and extract the official source release referenced in the inventory;
its aircraft, engine and systems files must match the complete frozen asset roster.
The Python wheel and release data are separate inputs. Run from the implementation
checkout recorded in the result, with its `src` on `PYTHONPATH`:

```sh
python -m glassbox.experimental.jsbsim_onboarding run \
  --root /absolute/path/to/jsbsim-1.3.1 \
  --out /absolute/path/to/new-evidence-directory
python -m glassbox.experimental.jsbsim_onboarding verify \
  --root /absolute/path/to/jsbsim-1.3.1 \
  --out /absolute/path/to/evidence-directory \
  --seal-sha256 TRUSTED_RUN_JSON_SHA256
```

The trusted seal comes from the committed result record, not a newly calculated
hash of evidence that may have changed. Full verification requires the recorded
Python, NumPy, JSBSim native runtime and source identities. It checks every saved
array against fresh simulator execution, including exact dtype and bytes, and
independently recomputes accounting and response metrics. Unpaced execution clocks
are descriptive. A timeout is not a reproducible physical result, even if another
attempt also times out.

The next accuracy iteration must separately freeze operating conditions, data and
compute budgets, whole-recording and system holdouts, forecast/response metrics,
comparators and evidence criteria. This audit makes the onboarding gaps explicit;
it does not select a favorable subset on the basis of fitted model performance.
