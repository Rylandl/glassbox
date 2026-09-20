# Shared vehicle physics learner

This iteration builds one vehicle learning procedure with shared gravity and rigid-body integration. It learns six effective body accelerations from observed motion and arbitrary ordered commands, with causal command filtering and hidden memory. It uses no vehicle-family branch, mixer, actuator layout or physical-parameter input. Separate fitted weights per configuration are expected.

The named gap is vehicle mechanics and recursive planning horizon. The [frozen protocol](harness/shared-vehicle-physics-v1.json) fixes architecture, data, optimizer, four fresh fits, untouched +14M confirmation and the separate Dart trial. Complete implementation and all harness stages must be committed and source-bound before any scientific fit or simulation.

The research lifecycle provides fit, predict, update and save/load. Means can recurse to1.2s; error envelopes remain limited to the calibrated250ms horizon. Public v4 stays intact during qualification. Arbitrary Euclidean-channel synthetic results do not transfer to this vehicle-semantic contract.

Physical comparison uses identical retained Crazyflow/Cascade caches and v4 loss weights, two new candidate fits and no control refits. Acceptance requires joint forecast/response ratio at most0.97, each kind at most1.05, and retained broad regression and targeted angular guards. Absolute errors and missing truth remain visible.

Dart gets two matched fits from original8train/2calibration recordings, followed by one nominal trial perarm with identical real observed history, command seed, planner and task. Its historical1.47mm result used simulator-applied thrust and remains context. Replays, independent reduction and four alterations must verify evidence before any acceptance claim.

Status: protocol frozen on commit; implementation and evaluation pending.
