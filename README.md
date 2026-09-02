# Glassbox

Glassbox learns differentiable vehicle dynamics from recorded state and
actuator telemetry, keeps an explicit account of what the fitted model does not
know, and uses that belief for bounded online adaptation and model-predictive
control. It targets multirotors and fixed-wing aircraft flown by PX4, with
adapters for several published system-identification datasets and two
simulators.

The fitted artifact is a **dynamics belief**: a structured rigid-body model
with learned actuator lag and an optional compact residual, surrounded by the
prediction error observed on held-out flights, a rank-aware parameter
information matrix, a validity envelope, and update provenance. Evaluation,
online updates, and the NMPC controller all consume that one artifact.

## Install

```bash
uv sync --dev
```

Python 3.11 to 3.13. JAX runs on the CPU backend in float32 by default.
Optional extras: `crazyflow` for the pinned Crazyflow simulator,
`crazyflow-animation` to render its diagnostics, and `cascade` for the Cascade
fixed-wing plant, which installs from GitHub.

## Quickstart

Turn telemetry into canonical trajectories. From a PX4 ULog:

```bash
uv run glassbox-ulog extract flight.ulg flight.npz --rate 50
```

The [PX4 ULog guide](docs/guides/px4-ulog.md) covers fixed-wing logs,
ground-truth versus estimated states, and gap handling. The reference-corpus
commands (`glassbox-nanodrone`, `glassbox-x8`, `glassbox-epfl`, and
`glassbox-ulog prepare-arp` and `prepare-idf`) produce the same NPZ format.

Fit a belief on several flights. The final flight is held out for validation
and a no-lag ablation is written beside the model:

```bash
uv run glassbox-fit flights/*.npz \
  --model artifacts/belief.json --report artifacts/report.json
```

Use the belief for control:

```python
import jax.numpy as jnp

from glassbox import DynamicsBelief, NMPCController, SafetyEnvelope, TrackingTolerances
from glassbox.dynamics import hover_control

belief = DynamicsBelief.load("artifacts/belief.json")
controller = NMPCController(
    belief,
    TrackingTolerances.for_platform(belief.input_spec.vehicle.family),
    SafetyEnvelope(
        minimum_position_m=(-100.0, -100.0, -20.0),
        maximum_position_m=(100.0, 100.0, 100.0),
    ),
)

# NWU position and velocity, WXYZ quaternion, FLU body rates.
state = jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
result = controller.solve(
    state, controller.hold_reference(state), hover_control(belief.params)
)
command = result.command  # bounded even when result.command_usable is False
```

Update the belief from recent telemetry with `belief.update(trajectory)`. The
update is transactional: it proposes a bounded local move on early telemetry,
commits only when disjoint later telemetry improves, and otherwise returns the
original belief. See [dynamics beliefs](docs/concepts/dynamics-beliefs.md) and
[NMPC](docs/concepts/nmpc.md) for the full contracts.

## Layout

The package is a flat set of modules in `src/glassbox`. `import glassbox`
loads only the core; workflows, corpus adapters, and integrations are imported
on demand.

| Area | Modules |
| --- | --- |
| Core | `data`, `dynamics`, `model_family`, `geometry`, `px4_frames`, `linearization`, `covariance`, `identification`, `evaluation`, `runtime`, `model_io`, `synthetic`, `fixedwing_synthetic` |
| Beliefs | `belief`, `belief_io`, `parameter_evidence`, `parameter_prior`, `adaptation` |
| Control | `nmpc/`, `flight_supervisor`, `bootstrap_identification`, `online_bootstrap`, `angular_authority` |
| Telemetry and corpora | `px4_ulog`, `sitl_profile`, `fixedwing_sitl_profile`, `nanodrone_*`, `arp_reference`, `idf_reference`, `x8_*`, `epfl_*` |
| Workflows | `fit_cli`, `profile_benchmark`, `source_group_benchmark`, `predictive_ensemble`, `policy_selection`, `acceptance`, `fixedwing_gate`, `nmpc_benchmark`, `adaptation_benchmark`, `adaptive_recovery_benchmark`, `observation_*`, `streaming_evaluation` |
| Integrations | `integrations/px4`, `integrations/px4_nmpc_shadow`, `integrations/crazyflow*`, `integrations/cascade` |

The canonical state is 13 wide: NWU position and velocity, a WXYZ unit
quaternion from body to world, and FLU body rates. Commands are normalized
actuator inputs in the order the trajectory spec declares.

## Console scripts

| Script | Purpose |
| --- | --- |
| `glassbox-fit` | Fit a dynamics belief and report from trajectory NPZ files |
| `glassbox-prior` | Build a fleet parameter prior from fitted beliefs |
| `glassbox-ulog` | Inspect and extract PX4 ULogs; prepare the ARP and IDF-DS corpora |
| `glassbox-sitl-profile`, `glassbox-fixedwing-sitl-profile` | Fly and record scripted PX4 SITL maneuver profiles |
| `glassbox-synthetic`, `glassbox-fixedwing-synthetic` | Synthetic parameter-recovery demonstrations |
| `glassbox-profile-benchmark`, `glassbox-source-benchmark` | Maneuver-family and source-group holdout benchmarks |
| `glassbox-select-policy` | Cross-platform fitting-policy selection |
| `glassbox-ensemble-benchmark` | Predictive-ensemble uncertainty diagnostic |
| `glassbox-adaptation-benchmark`, `glassbox-adaptive-recovery` | Belief adaptation diagnostics, the second through NMPC recovery |
| `glassbox-nanodrone`, `glassbox-x8`, `glassbox-epfl` | Fetch, prepare, and evaluate the reference corpora; `glassbox-x8` also drives the Cascade validation |
| `glassbox-fixedwing-gate` | Cross-airframe fixed-wing development gate |
| `glassbox-nmpc-benchmark` | Closed-loop NMPC acceptance suite |
| `glassbox-px4-nmpc-shadow` | Passive NMPC shadow against live PX4 telemetry; never transmits |
| `glassbox-crazyflow-*` | Crazyflow prototype, bootstrap, throw, and animation diagnostics (`--extra crazyflow`) |

Every script prints its flags with `--help`.

## Tests

```bash
uv run pytest
```

The default suite runs in about a quarter of an hour on a laptop. Tests marked
`crazyflow` and `cascade` need those extras; deselect them with
`-m "not crazyflow and not cascade"` when the extras are not installed. The
PX4 SITL contract tests are opt-in:

```bash
GLASSBOX_RUN_PX4_SITL=1 uv run pytest -m px4_sitl tests/integration/test_px4_sitl.py -v
```

Lint with `uv run ruff check src tests`.

## Documentation

The [documentation index](docs/README.md) lists every page. The main entries:

- [Scope and current boundary](docs/scope.md)
- Concepts: [dynamics beliefs](docs/concepts/dynamics-beliefs.md),
  [NMPC](docs/concepts/nmpc.md),
  [flight supervisor](docs/concepts/flight-supervisor.md),
  [bootstrap identification](docs/concepts/bootstrap-identification.md),
  [predictive ensembles](docs/concepts/predictive-ensembles.md)
- Experiments: one page per corpus, simulator diagnostic, and gate under
  [`docs/experiments/`](docs/experiments/), each citing its recorded artifact
  in [`docs/results/`](docs/results/)
- [Literature review](docs/literature-review.md)

## Status

Both vehicle families have differentiable rollout, fitting, serialization, and
PX4 ULog ingestion. Fixed-wing short and medium rollouts transfer across
maneuver families, recording sessions, and two airframe configurations;
multirotor results are competitive with a published structured-residual
reference on one airframe and not yet consistently better than kinematic
persistence on another. Long rollouts on multi-minute sessions remain
unstable, parameters do not transfer between airframes, and every controller
result is a bounded simulation diagnostic rather than a flight-safety claim.
See [scope](docs/scope.md) for the full boundary.

## License and citation

Glassbox is released under the [Apache License, Version 2.0](LICENSE). The
reference flight corpora it can download (Nano-Quadrotor, ARP, IDF-DS,
Skywalker X8, EPFL TOPOPlane2) are distributed by their authors under their own
terms and are not covered by this license.

If you use Glassbox in academic work, please cite it. GitHub renders a
ready-made citation from [CITATION.cff](CITATION.cff); the BibTeX form is:

```bibtex
@software{lillibridge2026glassbox,
  author  = {Lillibridge, Ryland},
  title   = {Glassbox: telemetry-driven differentiable vehicle dynamics identification},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Rylandl/glassbox},
  license = {Apache-2.0}
}
```
