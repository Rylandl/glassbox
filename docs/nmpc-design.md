# Glassbox NMPC design and acceptance contract

## Scope

The NMPC layer consumes the same canonical rigid-body and control semantics as
the fitted dynamics model. It provides finite-horizon tracking and regulation;
terminal-pose docking, state estimation, reference generation, and hardware
mixing remain separate projects or adapters.

The dependency direction is deliberately one-way:

```text
model artifact -> runtime dynamics -> NMPC -> canonical command -> actuation adapter
                                      ^
                    state estimate, reference, exogenous forecast
```

Runtime and NMPC modules may depend on the canonical data, dynamics, and model
serialization layers. They must not import fitting CLIs, benchmark workflows,
policy selection, or research-only experiments.

## Runtime contract

A control-eligible artifact must carry:

- the typed prediction `TrajectorySpec`;
- its fixed integration/sample period;
- the training-derived body-velocity and angular-rate validity envelope;
- an optional prediction horizon certified by named external evidence;
- the complete latent applied-control state layout; and
- an actionable command mapping with finite command bounds.

`normalized_command` and `normalized_generalized_command` are directly
actionable. Measured rotor speed, generalized surface angle, and normalized
actuator output are observations of actuation, not commands. Such artifacts
require an explicit typed actuation map and otherwise fail closed.

## Initial solver policy

The first maintained solver is warm-started direct shooting. It optimizes a
bounded sequence of command blocks, expands them over fixed model integration
steps, and differentiates the complete rollout with JAX. The normal interface
exposes physical tracking tolerances and a safety envelope rather than raw
state/control weight matrices. Solver iteration limits, line-search policy,
regularization, and control-block policy are maintainer-owned defaults.

Rigid-body error has 12 local coordinates: position, velocity, the shortest
quaternion log-map rotation vector, and angular velocity. Quaternion component
subtraction is never a tracking metric. Command bounds are hard constraints;
command rate and model-validity terms use dimensionless physical
normalization. A solve returns the predicted state/latent/control traces,
initial and final costs, convergence status, timing, constraint diagnostics,
and the bounded fallback command selected on failure.

## Acceptance thresholds fixed before tuning

The following gates apply to the maintained synthetic scenario suite:

1. Every nominal solve and closed-loop sample is finite. No hard command bound
   may be violated beyond `1e-6` in normalized command coordinates.
2. Analytic objective gradients must agree with central finite differences to
   `2e-3` relative error on structured multirotor, structured fixed-wing, and
   structured-residual fixtures.
3. The equal-scenario geometric mean of normalized tracking RMS must be at
   most `0.80` of the declared non-optimizing baseline. No individual nominal
   scenario may exceed `1.05` times its baseline error.
4. Model-mismatch scenarios must remain finite and within the validity guard.
   Their aggregate normalized tracking RMS must be below the same baseline;
   no aggregate gain may hide an individual ratio above `1.10`.
5. Warm starting must not worsen the initial objective compared with the
   controller's cold-start policy for the same receding-horizon state.
6. Forced non-convergence, non-finite objectives, invalid estimates, and
   deadline failures must return an explicit failure status and a finite,
   bounded fallback command.
7. Post-JIT solve latency is reported as median, p90, and maximum on named
   hardware. Passing functional gates does not constitute a real-time claim.

The baseline and normalized error definition are recorded with each report.
Thresholds may only change in a reviewed contract revision made before the
candidate being judged is tuned.

## Validation progression

The maintained progression is synthetic truth, fitted-model mismatch, then PX4
SITL. Real hardware is excluded from this project goal. Synthetic coverage must
include multirotor hover, translation, and attitude; fixed-wing trim, altitude
or path tracking, turning, and a flap-enabled configuration. Both structured
and structured-residual models must remain differentiable through the runtime
and solver paths.

