# Five-checkpoint trajectory correction

This screen tested one change to the fast readout: correct the completed
250 ms trajectory at five evenly spaced physical checkpoints rather than only
at its endpoint. It did **not** solve the extrapolation problem. The candidate
is rejected as a model direction; production source is unchanged.

The direct readout, causal prefixes, feature map, physical integrator, online
normal equations and eight recordings/schedules matched the [endpoint
screen](readout-trajectory-head.md). The only candidate change was a stacked
75-dimensional residual of world velocity, body rate and orientation at
50, 100, 150, 200 and 250 ms. One dual Gauss–Newton correction followed each
direct solve. The screen made 126 post-prefix updates, scoring the first and
next frozen forecast origins. No vehicle metadata, family branch, prior fleet
fit or future observation was supplied.

| First-origin 250 ms case | Path velocity / rate | Endpoint-only velocity / rate | Direct velocity / rate |
| --- | ---: | ---: | ---: |
| Fixed wing 80 | 0.537 m/s / 3.980 rad/s | 0.214 / 0.342 | 2.450 / 8.956 |
| Fixed wing 81 | **123.748 / 153.341** | 1.902 / 2.138 | 8.163 / 22.063 |
| Truncated quad arm 135 | **11.247 / 164.616** | 11.032 / 163.717 | 7.873 / 102.736 |

The other quad first-origin rate errors were 6.057, 7.710 and 7.710 rad/s
for arm 115, arm 125 and quad change; the gentle paired quad remained low.
At the second origin, the path and earlier readouts were usually close, with
mixed small differences. Every first-origin forecast was finite, so a
nonfinite-only check would not detect the failure.

The completed-path objective decreased at every prefix correction. For fixed
wing 81 it fell from 0.282 to 0.137 in its defined normalized units, yet the
next 250 ms forecast became much worse. Its normalized coefficient change was
0.051, far below the unit trust radius. Adding checkpoints to the same
teacher-forced path did not constrain the learned recurrence outside that
path. This falsifies the simple hypothesis that endpoint-only fitting caused
the hard-quad loss. More correction objectives or backtracking guards are not
the next useful iteration.

Complete warm-update medians were 2.58–2.74 ms on fixed wing and about
7.45–7.55 ms on 10 ms quads. The full first-prefix and compilation cost is
saved separately. The committed three-fixture test covers 10/50 ms checkpoint
indices, analytic versus finite-difference Jacobians, the dual solution and
the causal update cursor. The authenticated saved-data verifier independently
recomputes all physical scores and correction algebra without refitting.
The exact [artifact and source index](readout-trajectory-path.json) makes this
negative result reproducible.

The next work should simplify how candidate model changes are measured. The
one-off evaluator duplicated over 500 lines for a model change of a few lines;
the eight-case run itself took seconds. A reusable runner with a fixed compact
result table and a small diagnostic subset should carry the common work,
followed by the broader frozen suite only for a promising model. The model
target remains one generic dynamics formulation, not a catalog of correction
schemes.
