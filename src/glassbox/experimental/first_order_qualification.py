"""One frozen ftol=0 comparison using the existing optimizer evidence harness."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

from . import quasi_newton_qualification as parent
from . import solver_budget as base
from .qualification import arrays, digest, same
from .quasi_newton import QuasiNewtonSolver

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "docs/harness/solver-first-order-v1.json"
PLAN_SHA256 = "4d7a341601183fa3cecb5321ffb4df9b94aeeebdf3219e3a089f2350cde062a4"
check_environment = parent.check_environment
CORE_EDITS = (
    (
        "    objective_gradient, seed_blocks, seed_value, seed_gradient, policy\n",
        "    objective_gradient, seed_blocks, seed_value, seed_gradient, policy, *, _ftol=None\n",
        1,
    ),
    (
        "    seed = np.asarray(seed_blocks)\n",
        "    seed = np.asarray(seed_blocks)\n"
        "    if _ftol is not None and (type(_ftol) is not float or _ftol != 0.0):\n"
        "        raise ValueError(\n"
        '            "only the frozen zero relative-improvement override is allowed"\n'
        "        )\n",
        1,
    ),
    (
        "                ftol=policy.relative_improvement_tolerance,\n",
        "                ftol=policy.relative_improvement_tolerance if _ftol is None else _ftol,\n",
        1,
    ),
    (
        '    """Replace only optimization; inherit seeds, dynamics and result checks."""\n\n',
        '    """Replace only optimization; inherit seeds, dynamics and result checks."""\n\n'
        "    _ftol = None\n\n",
        1,
    ),
    (
        "            objective_gradient, blocks, value, gradient, self.policy\n",
        "            objective_gradient, blocks, value, gradient, self.policy, _ftol=self._ftol\n",
        1,
    ),
)
CONTEXT_EDITS = (
    ("import json\n", "import json\nimport sys\n", 1),
    (
        "def run(artifacts, output):\n",
        "def run(artifacts, output, *, _experiment=None):\n"
        "    experiment = _experiment or sys.modules[__name__]\n",
        1,
    ),
    (
        "def verify(directory):\n",
        "def verify(directory, *, _experiment=None):\n"
        "    experiment = _experiment or sys.modules[__name__]\n",
        1,
    ),
    (
        "    plan, raw = frozen_plan()\n",
        "    plan, raw = experiment.frozen_plan()\n",
        2,
    ),
    (
        "    environment = check_environment(plan)\n",
        "    environment = experiment.check_environment(plan)\n",
        2,
    ),
    (
        "    trials, work, result = probe(plan, inputs)\n",
        "    trials, work, result = experiment.probe(plan, inputs)\n",
        1,
    ),
    (
        "    same(summary, report_from_inputs(plan, trials, inputs, work))\n",
        "    same(summary, experiment.report_from_inputs(plan, trials, inputs, work))\n",
        1,
    ),
    (
        "    fresh, fresh_work, result = probe(plan, inputs)\n",
        "    fresh, fresh_work, result = experiment.probe(plan, inputs)\n",
        1,
    ),
)


def frozen_plan():
    raw = PLAN_PATH.read_bytes()
    if digest(raw) != PLAN_SHA256:
        raise ValueError("first-order plan differs from frozen source")
    plan = json.loads(raw)
    for name, expected in plan["source_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"first-order inherited source changed: {name}")
    for name, edits in (
        ("quasi_newton.py", CORE_EDITS),
        ("quasi_newton_qualification.py", CONTEXT_EDITS),
    ):
        path = "src/glassbox/experimental/" + name
        source = (ROOT / path).read_text()
        for before, after, count in edits:
            if source.count(after) != count:
                raise ValueError(f"first-order private hook differs: {name}")
            source = source.replace(after, before)
        if digest(source.encode()) != plan["allowed_edits"][path]["baseline_sha256"]:
            raise ValueError(f"first-order source changed beyond private hook: {name}")
    return plan, raw


class FirstOrderSolver(QuasiNewtonSolver):
    """Only the frozen private backend stopping override differs."""

    _ftol = 0.0


def historical_work_parity(actual, expected):
    same(actual, expected)
    for key in parent.WORK_ARRAYS:
        np.testing.assert_array_equal(actual[key], expected[key])


def paired_solve(
    model,
    policy,
    state,
    reference,
    previous_command,
    *,
    warm_start=None,
    baseline_check=None,
    budget_check=None,
    historical_check=None,
    historical_work_check=None,
    work_sink=None,
):
    args = (model, policy, state, reference, previous_command)
    previous_work = []
    previous = parent.paired_solve(
        *args,
        warm_start=warm_start,
        baseline_check=baseline_check,
        historical_check=budget_check,
        work_sink=previous_work.append,
    )["candidate"]
    if historical_check is not None:
        historical_check(previous)
    if historical_work_check is not None:
        historical_work_check(previous_work[0])
    solver = FirstOrderSolver(model, base.candidate_policy(policy))
    candidate = solver.solve(state, reference, previous_command, warm_start=warm_start)
    np.testing.assert_allclose(
        previous.diagnostics.initial_objective,
        candidate.diagnostics.initial_objective,
        **base.SCORE_TOLERANCE,
    )
    same(previous.diagnostics.warm_start_used, candidate.diagnostics.warm_start_used)
    if work_sink is not None:
        work_sink(dict(solver.last_work))
    return dict(baseline=previous, candidate=candidate)


def report_from_inputs(plan, trials, inputs, work):
    result = parent.report_from_inputs(plan, trials, parent.task_inputs(inputs), work)
    result["plan_sha256"] = PLAN_SHA256

    def summary(selected):
        messages = [row["backend_message"] for row in selected]
        gradient_exits = [
            row
            for row in selected
            if row["backend_message"].startswith(
                "CONVERGENCE: NORM OF PROJECTED GRADIENT"
            )
        ]
        return dict(
            origins=len(selected),
            raw_messages={
                message: messages.count(message) for message in sorted(set(messages))
            },
            nonpositive_relative_decrease_count=sum(
                message.startswith("CONVERGENCE: RELATIVE REDUCTION")
                for message in messages
            ),
            raw_projected_gradient_exit_count=len(gradient_exits),
            raw_gradient_exit_with_nonstationary_return_count=sum(
                row["independent_projected_gradient_inf_norm"] is not None
                and row["independent_projected_gradient_inf_norm"] > 0.002
                for row in gradient_exits
            ),
        )

    result["raw_backend_termination"] = dict(
        meaning=(
            "At ftol=0 a relative-reduction exit means nonpositive relative decrease, "
            "not proven roundoff. Raw backend exits and independent stationarity "
            "of the retained best point are separate measurements."
        ),
        pooled=summary(work),
        per_seed=[
            dict(seed=seed, **summary([row for row in work if row["seed"] == seed]))
            for seed in plan["selection"]["seeds"]
        ],
    )
    return result


def probe(plan, inputs):
    budget_inputs = parent.task_inputs(inputs)
    historical, budget = (
        [
            arrays(source[f"trial-{i}/paired.npz"])
            for i in range(len(plan["selection"]["seeds"]))
        ]
        for source in (inputs, budget_inputs)
    )
    previous_work = json.loads(inputs["work.json"])
    parent.validate_work(json.loads(inputs["manifest.json"]), historical, previous_work)
    base.validate_arrays(json.loads(budget_inputs["manifest.json"]), budget)
    work = []
    count = len(plan["selection"]["origins"])

    def solve(*args, **kwargs):
        index = len(work)
        repetition, row = divmod(index, count)
        seed, origin = (
            plan["selection"]["seeds"][repetition],
            plan["selection"]["origins"][row],
        )
        return paired_solve(
            *args,
            **kwargs,
            budget_check=partial(
                parent.historical_parity, trial=budget[repetition], row=row
            ),
            historical_check=partial(
                parent.historical_parity, trial=historical[repetition], row=row
            ),
            historical_work_check=lambda values: historical_work_parity(
                dict(seed=seed, origin=origin, **values), previous_work[index]
            ),
            work_sink=lambda values: work.append(
                dict(seed=seed, origin=origin, **values)
            ),
        )

    trials, _ = base.probe(plan, parent.task_inputs(budget_inputs), _solve_pair=solve)
    return trials, work, report_from_inputs(plan, trials, inputs, work)


def run(artifacts, output):
    return parent.run(artifacts, output, _experiment=sys.modules[__name__])


def verify(directory):
    return parent.verify(directory, _experiment=sys.modules[__name__])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--artifacts", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("verify")
    replay.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = (
        verify(args.directory)
        if args.command == "verify"
        else run(args.artifacts, args.output)
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
