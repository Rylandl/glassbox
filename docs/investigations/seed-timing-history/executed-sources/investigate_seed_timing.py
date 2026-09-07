"""Fixed, paired replay of the saved tick10 seed-stage deadline failure."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import subprocess
import time
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from investigate_fast_suffix_runtime import solve_waveform
from investigate_feedback_recovery import (
    RecoverySolver,
    SeedRequest,
    json_finite,
    prewarm,
)
from investigate_feedback_suffix import checker, describe, digest
from investigate_sqp_recovery import DEFAULT_WORK_ESTIMATES
from investigate_terminal_suffix import fingerprints

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController
from glassbox.core.synthetic import resting_state

DESIGN = {
    "absolute_tick": 10,
    "deadline_s": 0.02,
    "blocks": 64,
    "order_per_block": ["baseline", "traced", "traced", "baseline"],
    "measured_requests": 256,
    "prewarm_requests": 2,
    "deadline_free_parity_requests": 2,
    "request": "same saved state, actuator state, previous command and incoming unshifted forecast9 every time",
    "no_plant_steps": True,
    "gc_policy": "unchanged; collection callbacks observed in both arms",
    "blas_policy": "unchanged",
    "traced_materialization": "np.asarray(array) at existing seed materialization points, followed by np.asarray(host, dtype=float); no additional readiness fences",
}


class Recorder:
    """In-memory wall/process/thread clocks; intervals can contain worker work."""

    def __init__(self):
        self.origin = self.clock()
        self.events = []
        self.gc_events = []

    @staticmethod
    def clock():
        return time.perf_counter_ns(), time.process_time_ns(), time.thread_time_ns()

    def stamp(self):
        return [(a - b) * 1e-9 for a, b in zip(self.clock(), self.origin)]

    def mark(self, label):
        self.events.append([label, *self.stamp()])

    def invoke(self, label, function, *args, **kwargs):
        self.mark(label + ".start")
        try:
            return function(*args, **kwargs)
        finally:
            self.mark(label + ".end")

    def to_host(self, value, *, dtype):
        # The first conversion waits where the original float conversion waited.
        # The second isolates the host dtype copy without waiting on extra leaves.
        array = self.invoke("seed.materialize", np.asarray, value)
        return self.invoke("seed.float_conversion", np.asarray, array, dtype=dtype)

    def gc_callback(self, phase, info):
        self.gc_events.append([phase, *self.stamp(), dict(info)])

    def totals(self):
        starts, totals = {}, {}
        for label, *stamp in self.events:
            name, edge = label.rsplit(".", 1)
            if edge == "start":
                starts.setdefault(name, []).append(stamp)
            elif starts.get(name):
                start = starts[name].pop()
                total = totals.setdefault(name, [0.0, 0.0, 0.0])
                for i in range(3):
                    total[i] += stamp[i] - start[i]
        return totals


@contextmanager
def observe(solver, recorder):
    """Restore wrappers even on an abort; kernel wrappers measure dispatch only."""
    names = (
        "set_seed",
        "_reject_invalid_request",
        "_initial_latent",
        "_exogenous_forecast",
        "_cold_blocks",
        "_seed_plan",
        "_optimize_plan",
        "_solved_result",
        "_failure_result",
        "evaluate",
        "linearize",
        "finalize",
    )
    originals = {name: getattr(solver, name) for name in names}
    prior = getattr(solver, "_seed_observer", None)
    try:
        solver._seed_observer = recorder
        for name, original in originals.items():
            label = name + (".dispatch" if name in names[-3:] else "")

            def measured(*args, _label=label, _original=original, **kwargs):
                return recorder.invoke(_label, _original, *args, **kwargs)

            setattr(solver, name, measured)
        yield
    finally:
        solver._seed_observer = prior
        for name, original in originals.items():
            setattr(solver, name, original)


@contextmanager
def observe_request(solver):
    """Opt-in observation for the full recovery history, including early aborts."""
    recorder = Recorder()
    gc.callbacks.append(recorder.gc_callback)
    try:
        with observe(solver, recorder):
            yield recorder
    finally:
        gc.callbacks.remove(recorder.gc_callback)


def caller_trace(recorder, started, elapsed):
    origin = recorder.origin[0] * 1e-9
    interval = [started - origin, started + elapsed - origin]
    return {
        "event_columns": ["label", "wall_s", "process_cpu_s", "thread_cpu_s"],
        "phase_total_columns": ["wall_s", "process_cpu_s", "thread_cpu_s"],
        "caller_interval_s": interval,
        "events": recorder.events,
        "phase_totals_s": recorder.totals(),
        "gc_events": recorder.gc_events,
        "gc_events_in_caller": [
            event
            for event in recorder.gc_events
            if interval[0] <= event[1] <= interval[1]
        ],
    }


def saved_request(runtime):
    report = json.loads((runtime / "report.json").read_text())
    case = report["cases"]["cold_original_kick"]
    path = runtime / case["trace_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["trace_sha256"]
    with np.load(path) as trace:
        state = trace["states"][10].copy()
        latent = trace["latent_states"][10].copy()
        seed = trace["seed_waveforms"][10].copy()
        # The old report exported commands through Python lists into float64
        # NPZ arrays. Recover the original JAX dtype, verifying a lossless cast.
        incoming = trace["forecast_9"].astype(seed.dtype)
        np.testing.assert_array_equal(incoming, trace["forecast_9"])
        previous = incoming[0].copy()
    failed = case["requests"][-1]
    passed = report["cases"]["cold_original"]["requests"][10]
    assert failed["absolute_tick"] == 10 and not failed["applied"]
    assert passed["applied"]
    hashes = {}
    for key, value in (
        ("state", state),
        ("latent", latent),
        ("previous_command", previous),
        ("seed", seed),
    ):
        hashes[key + "_sha256"] = digest(value)
        assert failed[key + "_sha256"] == hashes[key + "_sha256"]
        assert passed[key + "_sha256"] == hashes[key + "_sha256"]
    np.testing.assert_array_equal(np.concatenate((incoming[1:], incoming[-1:])), seed)
    return (state, latent, previous, incoming, seed), hashes


def summarize(rows):
    result = {}
    for arm in DESIGN["order_per_block"][:2]:
        selected = [r for r in rows if r["arm"] == arm]
        fractions = np.array([r["caller_budget_fraction"] for r in selected])
        result[arm] = {
            "requests": len(selected),
            "accepted": sum(r["accepted"] for r in selected),
            "status_counts": dict(Counter(r["status"] for r in selected)),
            "message_counts": dict(Counter(r["message"] for r in selected)),
            "gc_requests": sum(bool(r["gc_events_in_caller"]) for r in selected),
            "caller_budget_fraction": {
                "min": float(fractions.min()),
                "median": float(np.median(fractions)),
                "p95": float(np.quantile(fractions, 0.95)),
                "max": float(fractions.max()),
            },
        }
    names = sorted({name for r in rows for name in r["phase_totals_s"]})
    result["traced_phase_wall_budget_fractions"] = {
        name: {
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }
        for name in names
        if (
            values := [
                r["phase_totals_s"][name][0] / DESIGN["deadline_s"]
                for r in rows
                if name in r["phase_totals_s"]
            ]
        )
    }
    result["traced_to_baseline_median_ratio"] = (
        result["traced"]["caller_budget_fraction"]["median"]
        / result["baseline"]["caller_budget_fraction"]["median"]
    )
    return result


def run(fixtures, runtime, output):
    # Refuse to erase a previous result. Predeclare count/order before setup.
    output.mkdir(parents=True, exist_ok=False)
    (output / "design.json").write_text(json.dumps(DESIGN, indent=2) + "\n")
    arrays, hashes = saved_request(runtime)
    state, latent, previous, incoming, seed = [jnp.asarray(x) for x in arrays]
    controller = NMPCController(DynamicsBelief.load(fixtures / "rich-belief.json"))
    plan = controller.plan
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    with np.load(fixtures / "horizon-shift-states.npz") as fixture:
        np.testing.assert_array_equal(reference.states, fixture["reference_states"])
    provenance = fingerprints(fixtures)
    prior = json.loads((runtime / "report.json").read_text())
    assert provenance["fixture_sha256"] == prior["fixture_sha256"]
    solver = RecoverySolver(plan, seed, work_estimates=DEFAULT_WORK_ESTIMATES)
    check = checker(plan, reference)
    request = SeedRequest(previous, incoming)
    prewarm(solver, state, latent, previous, request, reference, check)
    # A numerical parity check has no deadline branch and applies no command.
    plain = solve_waveform(
        solver, request, state, reference, previous, latent_state=latent
    )
    with observe(solver, Recorder()):
        detailed = solve_waveform(
            solver, request, state, reference, previous, latent_state=latent
        )
    assert plain.command_usable and detailed.command_usable
    assert plain.status == detailed.status
    for name in (
        "command",
        "predicted_commands",
        "predicted_states",
        "predicted_latent_states",
    ):
        np.testing.assert_array_equal(getattr(plain, name), getattr(detailed, name))
    assert plain.diagnostics.final_objective == detailed.diagnostics.final_objective
    assert plain.nonlinear_feasibility == detailed.nonlinear_feasibility
    configuration = io.StringIO()
    with redirect_stdout(configuration):
        np.show_config()
        scipy.show_config()
    report = {
        "diagnostic_only": True,
        "design": DESIGN,
        "input_hashes": hashes,
        "input_report_sha256": hashlib.sha256(
            (runtime / "report.json").read_bytes()
        ).hexdigest(),
        "work_estimates": asdict(DEFAULT_WORK_ESTIMATES),
        "platform": platform.platform(),
        "versions": {
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "x64": bool(jax.config.jax_enable_x64),
        "numeric_configuration": configuration.getvalue(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "gc_enabled": gc.isenabled(),
        "gc_thresholds": gc.get_threshold(),
        "deadline_free_output_parity_bitwise": True,
        "event_columns": ["label", "wall_s", "process_cpu_s", "thread_cpu_s"],
        "phase_total_columns": ["wall_s", "process_cpu_s", "thread_cpu_s"],
        "limitations": [
            "Identical-input replay does not reproduce the prior process history or host scheduling.",
            "No plant steps or recovery claim; every request resets from the recorded incoming waveform, including after rejection.",
            "Baseline disables observation wrappers but retains dormant research seed hooks and the same GC callback.",
            "Tracing allocates records and consumes deadline budget; dtype conversion is split only in the traced seed.",
            "Kernel dispatch is not execution time; materialization includes waiting and host access. No extra prediction leaves are synchronized.",
            "Wall, process and caller-thread clocks are sampled sequentially; process CPU includes native workers and neither CPU clock identifies host descheduling alone.",
            "Phase totals are inclusive and nested; do not sum the seed method with its dispatch, materialization and host subphases.",
            "Compilation logging is enabled around all measured requests; input hashing, independent checking and serialization are outside each timer.",
        ],
        **provenance,
        "requests": [],
    }
    forecasts = {}
    for block in range(DESIGN["blocks"]):
        for arm in DESIGN["order_per_block"]:
            recorder = Recorder()
            solver.reports.clear()
            phase_times = {}
            gc.callbacks.append(recorder.gc_callback)
            try:
                with jax.log_compiles(True):
                    if arm == "traced":
                        with observe(solver, recorder):
                            started = recorder.clock()
                            result = solve_waveform(
                                solver,
                                request,
                                state,
                                reference,
                                previous,
                                latent_state=latent,
                                deadline_s=DESIGN["deadline_s"],
                                phase_times=phase_times,
                            )
                            ended = recorder.clock()
                    else:
                        started = recorder.clock()
                        result = solve_waveform(
                            solver,
                            request,
                            state,
                            reference,
                            previous,
                            latent_state=latent,
                            deadline_s=DESIGN["deadline_s"],
                            phase_times=phase_times,
                        )
                        ended = recorder.clock()
            finally:
                gc.callbacks.remove(recorder.gc_callback)
            elapsed = (ended[0] - started[0]) * 1e-9
            interval = [(s[0] - recorder.origin[0]) * 1e-9 for s in (started, ended)]
            row = describe(solver, result, state, latent, previous, check, 0)
            np.testing.assert_array_equal(solver.seed_commands, seed)
            assert not solver.startup_request
            if row["optimizer"] is not None:
                assert row["optimizer"]["iteration_budget"] == 2
            index = len(report["requests"])
            forecasts[f"forecast_{index}"] = np.asarray(result.predicted_commands)
            row.pop("commands", None)
            row.pop("state", None)
            row.update(
                index=index,
                block=block,
                arm=arm,
                caller_elapsed_s=elapsed,
                caller_interval_s=interval,
                caller_cpu_s=[(a - b) * 1e-9 for a, b in zip(ended[1:], started[1:])],
                caller_budget_fraction=elapsed / DESIGN["deadline_s"],
                accepted=result.command_usable and elapsed < DESIGN["deadline_s"],
                applied=False,
                request_elapsed_s=result.diagnostics.solve_time_s,
                wrapper_phase_times=phase_times,
                phase_totals_s=recorder.totals(),
                events=recorder.events,
                gc_events=recorder.gc_events,
                gc_events_in_caller=[
                    event
                    for event in recorder.gc_events
                    if interval[0] <= event[1] <= interval[1]
                ],
                seed_linearization_time_s=solver._seed_linearization_time_s,
            )
            report["requests"].append(row)
        if (block + 1) % 16 == 0:
            print(
                f"Completed {4 * (block + 1)} / {DESIGN['measured_requests']} fixed replays",
                flush=True,
            )
    assert len(report["requests"]) == DESIGN["measured_requests"]
    report["summary"] = summarize(report["requests"])
    np.savez_compressed(output / "forecasts.npz", **forecasts)
    report["forecasts_sha256"] = hashlib.sha256(
        (output / "forecasts.npz").read_bytes()
    ).hexdigest()
    root = Path(__file__).resolve().parents[1]
    report["baseline_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    archive = output / "executed-sources"
    archive.mkdir()
    for name in (
        Path(__file__).name,
        "investigate_feedback_recovery.py",
        "investigate_feedback_suffix.py",
        "investigate_fast_suffix.py",
        "investigate_fast_suffix_runtime.py",
        "investigate_sqp_recovery.py",
    ):
        source = Path(__file__).with_name(name).read_bytes()
        (archive / name).write_bytes(source)
        report["source_sha256"][f"scripts/{name}"] = hashlib.sha256(source).hexdigest()
    (output / "report.json").write_text(
        json.dumps(json_finite(report), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.fixtures, args.runtime, args.output)
