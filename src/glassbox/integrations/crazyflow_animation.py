"""Render the no-prior Crazyflow bootstrap diagnostic as an annotated video."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.integrations.crazyflow import CrazyflowPlant, CrazyflowPlantConfig
from glassbox.integrations.crazyflow_bootstrap import (
    CrazyflowBootstrapTrace,
    run_crazyflow_bootstrap_trial,
)
from glassbox.integrations.crazyflow_telemetry import tilt_rad as _tilt_rad
from glassbox.integrations.crazyflow_throw import (
    CrazyflowThrowTrace,
    run_crazyflow_throw_trial,
)
from glassbox.integrations.crazyflow_throw_study import (
    ARM_DISPLAY_NAMES,
    CRAZYFLOW_THROW_STUDY_CASES,
    STUDY_CONTROL_MODELS,
    CrazyflowStudyTrace,
    run_throw_study_render_trial,
)

_NAVY = (8, 17, 28)
_MINT = (78, 232, 184)
_CYAN = (92, 210, 255)
_AMBER = (255, 190, 72)
_RED = (255, 100, 105)
_WHITE = (241, 247, 250)
_MUTED = (158, 178, 190)


@dataclass(frozen=True)
class CrazyflowAnimationConfig:
    """Deterministic output and presentation settings."""

    width: int = 1280
    height: int = 720
    frames_per_second: int = 30
    evidence_playback_s: float = 1.6
    targeted_playback_s: float = 0.7
    recovery_playback_s: float = 2.2

    def __post_init__(self) -> None:
        if self.width < 640 or self.height < 360:
            raise ValueError("animation dimensions must be at least 640x360")
        if self.width % 2 or self.height % 2:
            raise ValueError("animation dimensions must be even for H.264")
        if self.frames_per_second < 1:
            raise ValueError("frames_per_second must be positive")
        for name in (
            "evidence_playback_s",
            "targeted_playback_s",
            "recovery_playback_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class _StoryMoment:
    phase: str
    status: str
    detail: str
    simulation_time_s: float
    progress: float


def _hold(
    *,
    phase: str,
    status: str,
    detail: str,
    simulation_time_s: float,
    progress: float,
    duration_s: float,
    frames_per_second: int,
) -> list[_StoryMoment]:
    return [
        _StoryMoment(
            phase=phase,
            status=status,
            detail=detail,
            simulation_time_s=simulation_time_s,
            progress=progress,
        )
        for _ in range(max(1, round(duration_s * frames_per_second)))
    ]


def _sweep(
    *,
    phase: str,
    status: str,
    detail: str,
    start_time_s: float,
    end_time_s: float,
    start_progress: float,
    end_progress: float,
    duration_s: float,
    frames_per_second: int,
) -> list[_StoryMoment]:
    count = max(2, round(duration_s * frames_per_second))
    return [
        _StoryMoment(
            phase=phase,
            status=status,
            detail=detail,
            simulation_time_s=float(time_s),
            progress=float(progress),
        )
        for time_s, progress in zip(
            np.linspace(start_time_s, end_time_s, count),
            np.linspace(start_progress, end_progress, count),
        )
    ]


def _storyboard(
    trace: CrazyflowBootstrapTrace,
    config: CrazyflowAnimationConfig,
) -> list[_StoryMoment]:
    fps = config.frames_per_second
    evidence_start = float(trace.evidence_timestamps_s[0])
    provisional_time = float(
        trace.evidence_timestamps_s[trace.provisional_interval_count]
    )
    evidence_end = float(trace.evidence_timestamps_s[-1])
    recovery_start = float(trace.recovery_timestamps_s[0])
    recovery_end = float(trace.recovery_timestamps_s[-1])
    moments = _hold(
        phase="evidence",
        status="NO AIRFRAME PRIOR",
        detail="Only four bounded motor channels and measured state are known",
        simulation_time_s=evidence_start,
        progress=0.02,
        duration_s=0.4,
        frames_per_second=fps,
    )
    moments += _sweep(
        phase="evidence",
        status="BOUNDED EXPLORATION",
        detail="Learning each motor's local effect from applied input",
        start_time_s=evidence_start,
        end_time_s=provisional_time,
        start_progress=0.03,
        end_progress=0.29,
        duration_s=config.evidence_playback_s,
        frames_per_second=fps,
    )
    moments += _hold(
        phase="evidence",
        status="PROVISIONAL FIT REJECTED",
        detail="Yaw prediction is worse than its nuisance-only baseline",
        simulation_time_s=provisional_time,
        progress=0.36,
        duration_s=0.5,
        frames_per_second=fps,
    )
    moments += _sweep(
        phase="evidence",
        status="TARGETING WEAK YAW EVIDENCE",
        detail="Four symmetric pulses use the provisional learned direction",
        start_time_s=provisional_time,
        end_time_s=evidence_end,
        start_progress=0.37,
        end_progress=0.49,
        duration_s=config.targeted_playback_s,
        frames_per_second=fps,
    )
    moments += _hold(
        phase="evidence",
        status="LOCAL MODEL READY",
        detail="4 input directions + 3 angular outputs supported in 0.56 s",
        simulation_time_s=evidence_end,
        progress=0.55,
        duration_s=0.5,
        frames_per_second=fps,
    )
    moments += _hold(
        phase="recovery",
        status="SEPARATE STABILIZATION TRIAL",
        detail="Reset to throw-like velocity, tilt, and body rates",
        simulation_time_s=recovery_start,
        progress=0.60,
        duration_s=0.35,
        frames_per_second=fps,
    )
    moments += _sweep(
        phase="recovery",
        status="LEARNED RATE + VELOCITY ARREST",
        detail="Bounded motor inputs come directly from the fitted local effects",
        start_time_s=recovery_start,
        end_time_s=recovery_end,
        start_progress=0.61,
        end_progress=0.93,
        duration_s=config.recovery_playback_s,
        frames_per_second=fps,
    )
    moments += _hold(
        phase="recovery",
        status="STABILIZED IN SIMULATION",
        detail="Diagnostic result - not yet a throw-to-recover safety claim",
        simulation_time_s=recovery_end,
        progress=1.0,
        duration_s=0.75,
        frames_per_second=fps,
    )
    return moments


def _rank_four_time_s(trace: CrazyflowThrowTrace | CrazyflowStudyTrace) -> float:
    """First simulation time the flown command evidence reaches rank four.

    ``inf`` if it never does, which is honest: a dual-control arm can lose a
    case without ever earning full command evidence, and the trace does not
    pretend otherwise by falling back to a certification field that may not
    exist.
    """

    ranks = np.asarray(trace.command_evidence_ranks)
    hits = np.flatnonzero(ranks >= 4)
    return float("inf") if len(hits) == 0 else float(trace.timestamps_s[int(hits[0])])


def _throw_storyboard(
    trace: CrazyflowThrowTrace | CrazyflowStudyTrace,
    config: CrazyflowAnimationConfig,
    *,
    throw_only: bool = False,
) -> list[_StoryMoment]:
    """Play exact simulation time continuously, with no editorial holds.

    Three phases carry the frames — unpowered (motors and model both off),
    learning (model enable until command evidence reaches rank four), and
    learned control (after rank four) — with a release marker at time zero
    drawn separately by the overlay.  Nothing here reads a certification
    field, so a trace that never certifies or validates a belief (any
    dual-control arm, or the working-belief cascade) renders exactly the same
    way as one that does.
    """

    fps = config.frames_per_second
    release_time = float(trace.timestamps_s[0])
    enable_time = float(trace.timestamps_s[trace.model_enable_sample_index])
    rank_four_time = _rank_four_time_s(trace)
    terminal_time = enable_time if throw_only else float(trace.timestamps_s[-1])
    count = max(1, round((terminal_time - release_time) * fps))
    moments: list[_StoryMoment] = []
    for simulation_time in release_time + np.arange(count) / fps:
        simulation_time = float(simulation_time)
        if simulation_time < enable_time or throw_only:
            phase = "unpowered"
            status = "UNPOWERED THROW / SYSTEM OFF"
            detail = "Motors at zero - belief and controller are both disabled"
        elif simulation_time < rank_four_time:
            phase = "learning"
            status = "LEARNING WHILE ARRESTING"
            detail = "Command evidence has not yet reached rank four"
        else:
            phase = "learned"
            status = "LEARNED CONTROL"
            detail = "Command evidence spans all four channels"
        progress = (simulation_time - release_time) / max(
            terminal_time - release_time,
            1e-9,
        )
        moments.append(
            _StoryMoment(
                phase=phase,
                status=status,
                detail=detail,
                simulation_time_s=simulation_time,
                progress=float(progress),
            )
        )
    return moments


def _interpolate_sample(
    timestamps_s: np.ndarray,
    states: np.ndarray,
    commands: np.ndarray,
    time_s: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Interpolate a state-aligned trace, including a normalized quaternion."""

    clipped_time = float(np.clip(time_s, timestamps_s[0], timestamps_s[-1]))
    upper = int(np.searchsorted(timestamps_s, clipped_time, side="right"))
    if upper == 0:
        return states[0].copy(), commands[0].copy(), 0
    if upper >= len(timestamps_s):
        return states[-1].copy(), commands[-1].copy(), len(timestamps_s) - 1
    lower = upper - 1
    interval = timestamps_s[upper] - timestamps_s[lower]
    fraction = float((clipped_time - timestamps_s[lower]) / interval)
    state = (1.0 - fraction) * states[lower] + fraction * states[upper]
    first_quaternion = states[lower, 6:10]
    second_quaternion = states[upper, 6:10]
    if np.dot(first_quaternion, second_quaternion) < 0.0:
        second_quaternion = -second_quaternion
    quaternion = (1.0 - fraction) * first_quaternion + fraction * second_quaternion
    state[6:10] = quaternion / np.linalg.norm(quaternion)
    command = (1.0 - fraction) * commands[lower] + fraction * commands[upper]
    return state, command, lower


def _load_font(size: int, *, bold: bool = False, mono: bool = False) -> Any:
    from PIL import ImageFont

    if mono:
        candidates = (
            "/System/Library/Fonts/SFNSMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        )
    elif bold:
        candidates = (
            "/System/Library/Fonts/SFNS.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        )
    else:
        candidates = (
            "/System/Library/Fonts/SFNS.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_motor_bars(
    draw: Any,
    command: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    font: Any,
) -> None:
    bar_width = width // 7
    gap = bar_width
    for motor, value in enumerate(command):
        left = x + gap // 2 + motor * (bar_width + gap)
        top = y + round((1.0 - float(value)) * height)
        draw.rounded_rectangle(
            (left, y, left + bar_width, y + height),
            radius=4,
            fill=(*_MUTED, 40),
            outline=(*_MUTED, 110),
            width=1,
        )
        draw.rounded_rectangle(
            (left, top, left + bar_width, y + height),
            radius=4,
            fill=(*_AMBER, 220),
        )
        draw.text(
            (left + bar_width / 2, y + height + 8),
            str(motor + 1),
            font=font,
            fill=(*_MUTED, 255),
            anchor="ma",
        )


def _draw_normalized_curves(
    draw: Any,
    trace: CrazyflowBootstrapTrace,
    *,
    time_s: float,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    speed = np.linalg.norm(trace.recovery_states[:, 3:6], axis=1)
    rate = np.linalg.norm(trace.recovery_states[:, 10:13], axis=1)
    speed /= max(float(speed[0]), 1e-9)
    rate /= max(float(rate[0]), 1e-9)
    timestamps = trace.recovery_timestamps_s

    def points(values: np.ndarray) -> list[tuple[float, float]]:
        return [
            (
                x
                + width
                * float((time - timestamps[0]) / (timestamps[-1] - timestamps[0])),
                y + height * (1.0 - float(np.clip(value, 0.0, 1.0))),
            )
            for time, value in zip(timestamps, values)
        ]

    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=7,
        fill=(*_NAVY, 130),
        outline=(*_MUTED, 70),
        width=1,
    )
    draw.line(points(speed), fill=(*_CYAN, 255), width=3, joint="curve")
    draw.line(points(rate), fill=(*_AMBER, 255), width=3, joint="curve")
    fraction = float(
        np.clip((time_s - timestamps[0]) / (timestamps[-1] - timestamps[0]), 0.0, 1.0)
    )
    marker_x = x + round(width * fraction)
    draw.line((marker_x, y, marker_x, y + height), fill=(*_WHITE, 180), width=2)


def _draw_overlay(
    rgb_frame: np.ndarray,
    *,
    moment: _StoryMoment,
    state: np.ndarray,
    command: np.ndarray,
    sample_index: int,
    trace: CrazyflowBootstrapTrace,
    report: dict[str, Any],
    timeline_labels: tuple[tuple[float, str], ...] | None = None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(image, "RGBA")
    width, _ = image.size
    scale = width / 1280.0
    title_font = _load_font(max(20, round(31 * scale)), bold=True)
    status_font = _load_font(max(15, round(20 * scale)), bold=True)
    label_font = _load_font(max(11, round(14 * scale)), bold=True)
    body_font = _load_font(max(11, round(15 * scale)))
    mono_font = _load_font(max(11, round(15 * scale)), mono=True)
    small_font = _load_font(max(9, round(11 * scale)))

    margin = round(34 * scale)
    draw.rounded_rectangle(
        (margin, margin, round(790 * scale), round(145 * scale)),
        radius=round(14 * scale),
        fill=(*_NAVY, 215),
        outline=(*_WHITE, 35),
        width=1,
    )
    draw.text(
        (round(55 * scale), round(52 * scale)),
        "GLASSBOX / CRAZYFLOW",
        font=title_font,
        fill=(*_WHITE, 255),
    )
    status_color = _RED if "REJECTED" in moment.status else _MINT
    draw.text(
        (round(56 * scale), round(93 * scale)),
        moment.status,
        font=status_font,
        fill=(*status_color, 255),
    )
    draw.text(
        (round(56 * scale), round(120 * scale)),
        moment.detail,
        font=body_font,
        fill=(*_MUTED, 255),
    )

    panel_left = round(920 * scale)
    panel_right = width - margin
    panel_top = round(34 * scale)
    panel_bottom = round(626 * scale)
    draw.rounded_rectangle(
        (panel_left, panel_top, panel_right, panel_bottom),
        radius=round(14 * scale),
        fill=(*_NAVY, 218),
        outline=(*_WHITE, 35),
        width=1,
    )
    px = panel_left + round(22 * scale)
    draw.text(
        (px, round(57 * scale)),
        "MODEL STATUS",
        font=label_font,
        fill=(*_MUTED, 255),
    )
    model_status = "COLLECTING"
    model_color = _AMBER
    if "REJECTED" in moment.status:
        model_status = "NOT READY"
        model_color = _RED
    elif moment.progress >= 0.49:
        model_status = "LOCAL MODEL READY"
        model_color = _MINT
    draw.text(
        (px, round(82 * scale)),
        model_status,
        font=status_font,
        fill=(*model_color, 255),
    )

    speed = float(np.linalg.norm(state[3:6]))
    rate = float(np.linalg.norm(state[10:13]))
    tilt_deg = math.degrees(_tilt_rad(state))
    if moment.phase == "evidence":
        evidence_count = min(sample_index, len(trace.evidence_timestamps_s) - 1)
        draw.text(
            (px, round(132 * scale)),
            "EVIDENCE",
            font=label_font,
            fill=(*_MUTED, 255),
        )
        draw.text(
            (px, round(158 * scale)),
            f"{evidence_count:02d} / 28 intervals",
            font=mono_font,
            fill=(*_WHITE, 255),
        )
        draw.text(
            (px, round(200 * scale)),
            "YAW VALIDATION",
            font=label_font,
            fill=(*_MUTED, 255),
        )
        if sample_index < trace.provisional_interval_count:
            draw.text(
                (px, round(226 * scale)),
                "pending provisional fit",
                font=mono_font,
                fill=(*_MUTED, 255),
            )
        else:
            provisional_yaw = report["provisional_identification"]["validation"][
                "angular_improvement"
            ][2]
            final_yaw = report["identification"]["validation"]["angular_improvement"][2]
            yaw_value = (
                final_yaw
                if sample_index == len(trace.evidence_timestamps_s) - 1
                else provisional_yaw
            )
            yaw_color = _RED if yaw_value < 0.0 else _MINT
            draw.text(
                (px, round(226 * scale)),
                f"{yaw_value:+.3f} improvement",
                font=mono_font,
                fill=(*yaw_color, 255),
            )
        draw.text(
            (px, round(275 * scale)),
            "MEASURED MOTOR STATE",
            font=label_font,
            fill=(*_MUTED, 255),
        )
        _draw_motor_bars(
            draw,
            command,
            x=px,
            y=round(310 * scale),
            width=panel_right - px - round(18 * scale),
            height=round(122 * scale),
            font=small_font,
        )
        if moment.progress >= 0.49:
            hover = report["evaluation_only"]["estimated_hover_motor_command"]
            fit_ms = report["timing"]["fit_wall_time_s"] * 1e3
            draw.text(
                (px, round(490 * scale)),
                f"hover {hover:.4f}   fit {fit_ms:.2f} ms",
                font=mono_font,
                fill=(*_WHITE, 255),
            )
    else:
        recovery = report["velocity_attitude_rate_arrest"]
        initial_speed = recovery["initial_velocity_norm_m_s"]
        initial_rate = recovery["initial_angular_rate_norm_rad_s"]
        draw.text(
            (px, round(132 * scale)),
            "SPEED",
            font=label_font,
            fill=(*_CYAN, 255),
        )
        draw.text(
            (px, round(158 * scale)),
            f"{speed:5.2f} m/s   {100.0 * speed / initial_speed:4.0f}%",
            font=mono_font,
            fill=(*_WHITE, 255),
        )
        draw.text(
            (px, round(202 * scale)),
            "BODY RATE",
            font=label_font,
            fill=(*_AMBER, 255),
        )
        draw.text(
            (px, round(228 * scale)),
            f"{rate:5.2f} rad/s {100.0 * rate / initial_rate:4.0f}%",
            font=mono_font,
            fill=(*_WHITE, 255),
        )
        draw.text(
            (px, round(272 * scale)),
            "TILT",
            font=label_font,
            fill=(*_MUTED, 255),
        )
        draw.text(
            (px, round(298 * scale)),
            f"{tilt_deg:5.1f} deg",
            font=mono_font,
            fill=(*_WHITE, 255),
        )
        _draw_normalized_curves(
            draw,
            trace,
            time_s=moment.simulation_time_s,
            x=px,
            y=round(355 * scale),
            width=panel_right - px - round(20 * scale),
            height=round(105 * scale),
        )
        draw.text(
            (px, round(472 * scale)),
            "speed",
            font=small_font,
            fill=(*_CYAN, 255),
        )
        draw.text(
            (px + round(64 * scale), round(472 * scale)),
            "rate",
            font=small_font,
            fill=(*_AMBER, 255),
        )

    draw.text(
        (px, round(560 * scale)),
        "SIMULATION DIAGNOSTIC",
        font=label_font,
        fill=(*_WHITE, 230),
    )
    draw.text(
        (px, round(586 * scale)),
        "No flight-safety claim",
        font=body_font,
        fill=(*_MUTED, 255),
    )

    timeline_y = round(672 * scale)
    timeline_left = round(55 * scale)
    timeline_right = width - round(55 * scale)
    draw.line(
        (timeline_left, timeline_y, timeline_right, timeline_y),
        fill=(*_MUTED, 100),
        width=max(2, round(3 * scale)),
    )
    progress_x = timeline_left + round(
        moment.progress * (timeline_right - timeline_left)
    )
    draw.line(
        (timeline_left, timeline_y, progress_x, timeline_y),
        fill=(*_MINT, 255),
        width=max(3, round(5 * scale)),
    )
    draw.ellipse(
        (
            progress_x - round(6 * scale),
            timeline_y - round(6 * scale),
            progress_x + round(6 * scale),
            timeline_y + round(6 * scale),
        ),
        fill=(*_MINT, 255),
    )
    labels = (
        (
            (0.02, "NO PRIOR"),
            (0.34, "REJECT"),
            (0.52, "FIT"),
            (0.62, "RESET"),
            (0.92, "ARREST"),
        )
        if timeline_labels is None
        else timeline_labels
    )
    for position, label in labels:
        label_x = timeline_left + round(position * (timeline_right - timeline_left))
        draw.text(
            (label_x, timeline_y - round(23 * scale)),
            label,
            font=small_font,
            fill=(*_MUTED, 240),
            anchor="ms",
        )
    return np.asarray(image)


_PHASE_COLORS = {"unpowered": _AMBER, "learning": _CYAN, "learned": _MINT}


def _draw_throw_overlay(
    rgb_frame: np.ndarray,
    *,
    moment: _StoryMoment,
    state: np.ndarray,
    command: np.ndarray,
    sample_index: int,
    trace: CrazyflowThrowTrace | CrazyflowStudyTrace,
    throw_only: bool,
) -> np.ndarray:
    """Draw telemetry without changing the real-time throw playback.

    Every number comes from ``trace`` and the frame being drawn, never from a
    report or a literal constant, so a study arm that never certifies or even
    validates a belief (any dual-control arm, or the working-belief cascade)
    renders exactly the same overlay, honestly naming whatever arm flew it.
    """

    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(image, "RGBA")
    width, _ = image.size
    scale = width / 1280.0
    title_font = _load_font(max(20, round(31 * scale)), bold=True)
    status_font = _load_font(max(15, round(20 * scale)), bold=True)
    label_font = _load_font(max(11, round(14 * scale)), bold=True)
    body_font = _load_font(max(11, round(15 * scale)))
    mono_font = _load_font(max(11, round(15 * scale)), mono=True)
    small_font = _load_font(max(9, round(11 * scale)))
    margin = round(34 * scale)

    draw.rounded_rectangle(
        (margin, margin, round(835 * scale), round(145 * scale)),
        radius=round(14 * scale),
        fill=(*_NAVY, 215),
        outline=(*_WHITE, 35),
        width=1,
    )
    draw.text(
        (round(55 * scale), round(52 * scale)),
        "GLASSBOX / CRAZYFLOW",
        font=title_font,
        fill=(*_WHITE, 255),
    )
    phase_color = _PHASE_COLORS.get(moment.phase, _MINT)
    draw.text(
        (round(56 * scale), round(93 * scale)),
        moment.status,
        font=status_font,
        fill=(*phase_color, 255),
    )
    draw.text(
        (round(56 * scale), round(120 * scale)),
        moment.detail,
        font=body_font,
        fill=(*_MUTED, 255),
    )

    panel_left = round(920 * scale)
    panel_right = width - margin
    panel_top = round(34 * scale)
    panel_bottom = round(626 * scale)
    draw.rounded_rectangle(
        (panel_left, panel_top, panel_right, panel_bottom),
        radius=round(14 * scale),
        fill=(*_NAVY, 218),
        outline=(*_WHITE, 35),
        width=1,
    )
    px = panel_left + round(22 * scale)
    arm = getattr(trace, "arm", "certified")
    arm_label = ARM_DISPLAY_NAMES.get(arm, arm.upper())
    draw.text(
        (px, round(57 * scale)),
        "CONTROL MODEL",
        font=label_font,
        fill=(*_MUTED, 255),
    )
    draw.text(
        (px, round(82 * scale)),
        arm_label,
        font=status_font,
        fill=(*phase_color, 255),
    )

    speed = float(np.linalg.norm(state[3:6]))
    rate = float(np.linalg.norm(state[10:13]))
    tilt_deg = math.degrees(_tilt_rad(state))
    draw.text(
        (px, round(132 * scale)),
        "SPEED / BODY RATE",
        font=label_font,
        fill=(*_MUTED, 255),
    )
    draw.text(
        (px, round(158 * scale)),
        f"{speed:5.2f} m/s  {rate:5.2f} rad/s",
        font=mono_font,
        fill=(*_WHITE, 255),
    )
    draw.text(
        (px, round(198 * scale)),
        "TILT / WORKING UPDATES",
        font=label_font,
        fill=(*_MUTED, 255),
    )
    draw.text(
        (px, round(224 * scale)),
        f"{tilt_deg:5.1f} deg  {int(trace.working_interval_counts[sample_index]):4d}",
        font=mono_font,
        fill=(*_WHITE, 255),
    )
    live_rank = int(trace.command_evidence_ranks[sample_index])
    collective_command = float(np.mean(command))
    draw.text(
        (px, round(264 * scale)),
        "COMMAND RANK / COLLECTIVE",
        font=label_font,
        fill=(*_MUTED, 255),
    )
    draw.text(
        (px, round(290 * scale)),
        f"{live_rank}/4  {collective_command:.3f}",
        font=mono_font,
        fill=(*(_MINT if live_rank >= 4 else _WHITE), 255),
    )
    draw.text(
        (px, round(350 * scale)),
        "MEASURED MOTOR STATE",
        font=label_font,
        fill=(*_MUTED, 255),
    )
    _draw_motor_bars(
        draw,
        command,
        x=px,
        y=round(380 * scale),
        width=panel_right - px - round(18 * scale),
        height=round(84 * scale),
        font=small_font,
    )
    draw.text(
        (px, round(514 * scale)),
        f"t = {moment.simulation_time_s:4.2f} s",
        font=mono_font,
        fill=(*_WHITE, 255),
    )
    draw.text(
        (px, round(552 * scale)),
        "SIMULATION DIAGNOSTIC",
        font=label_font,
        fill=(*_WHITE, 230),
    )
    draw.text(
        (px, round(578 * scale)),
        "No flight-safety claim",
        font=body_font,
        fill=(*_MUTED, 255),
    )

    timeline_y = round(672 * scale)
    timeline_left = round(55 * scale)
    timeline_right = width - round(55 * scale)
    draw.line(
        (timeline_left, timeline_y, timeline_right, timeline_y),
        fill=(*_MUTED, 100),
        width=max(2, round(3 * scale)),
    )
    progress_x = timeline_left + round(
        moment.progress * (timeline_right - timeline_left)
    )
    draw.line(
        (timeline_left, timeline_y, progress_x, timeline_y),
        fill=(*_MINT, 255),
        width=max(3, round(5 * scale)),
    )
    draw.ellipse(
        (
            progress_x - round(6 * scale),
            timeline_y - round(6 * scale),
            progress_x + round(6 * scale),
            timeline_y + round(6 * scale),
        ),
        fill=(*_MINT, 255),
    )
    release_time = float(trace.timestamps_s[0])
    enable_time = float(trace.timestamps_s[trace.model_enable_sample_index])
    terminal_time = enable_time if throw_only else float(trace.timestamps_s[-1])
    span = max(terminal_time - release_time, 1e-9)
    markers: list[tuple[float, str]] = [(0.0, "RELEASE")]
    if throw_only:
        markers.append((1.0, f"{enable_time - release_time:.2f} s / OFF"))
    else:
        markers.append(((enable_time - release_time) / span, "ENABLE"))
        rank_four_time = _rank_four_time_s(trace)
        if math.isfinite(rank_four_time):
            markers.append(((rank_four_time - release_time) / span, "RANK 4"))
        certified_index = getattr(trace, "certified_belief_sample_index", None)
        if certified_index is not None:
            certified_time = float(trace.timestamps_s[certified_index])
            markers.append(((certified_time - release_time) / span, "CERTIFIED"))
    # Two markers close together in time (enable and an early rank four, most
    # often) would overlap at one text row, so alternate rows whenever markers
    # are closer than a label's own width can safely claim.
    row_gap = round(15 * scale)
    previous_x: float | None = None
    row = 0
    for position, label in markers:
        label_x = timeline_left + round(position * (timeline_right - timeline_left))
        if previous_x is not None and label_x - previous_x < round(70 * scale):
            row = 1 - row
        else:
            row = 0
        draw.text(
            (label_x, timeline_y - round(23 * scale) - row * row_gap),
            label,
            font=small_font,
            fill=(*_MUTED, 240),
            anchor="ms",
        )
        previous_x = label_x
    return np.asarray(image)


def _configure_camera(simulator: Any, *, lookat: np.ndarray, distance: float) -> None:
    if simulator.viewer is None:
        return
    camera = simulator.viewer.viewer.cam
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = 140.0
    camera.elevation = -18.0


def _render_plant_frame(
    plant: CrazyflowPlant,
    *,
    state: np.ndarray,
    command: np.ndarray,
    trail: np.ndarray,
    width: int,
    height: int,
    recovery: bool,
) -> np.ndarray:
    from crazyflow.sim.visualize import draw_line, draw_points

    plant.reset(state, applied_motor_thrust_fraction=command)
    simulator = plant._simulator
    if recovery:
        lookat = state[0:3]
        distance = 1.20
    else:
        lookat = np.asarray((0.05, 0.05, 1.95))
        distance = 0.78
    camera_config = None
    if simulator.viewer is None:
        camera_config = {
            "distance": distance,
            "azimuth": 140.0,
            "elevation": -18.0,
            "lookat": lookat,
        }
    _configure_camera(simulator, lookat=lookat, distance=distance)
    rgb = simulator.render(
        mode="rgb_array",
        width=width,
        height=height,
        cam_config=camera_config,
    )
    _configure_camera(simulator, lookat=lookat, distance=distance)
    if len(trail) > 1:
        draw_line(
            simulator,
            trail,
            rgba=np.asarray((*(_MINT if recovery else _AMBER), 255)) / 255.0,
            start_size=0.003,
            end_size=0.008,
        )
        draw_points(
            simulator,
            trail[:1],
            rgba=np.asarray((*_WHITE, 210)) / 255.0,
            size=0.004,
        )
        rgb = simulator.render(mode="rgb_array", width=width, height=height)
    return np.asarray(rgb, dtype=np.uint8)


def _start_encoder(
    output: Path,
    *,
    width: int,
    height: int,
    frames_per_second: int,
) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the animation")
    command = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(frames_per_second),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg did not expose a video input pipe")
    return process


def _finish_encoder(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is None:
        raise RuntimeError("ffmpeg video input pipe disappeared")
    process.stdin.close()
    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if return_code:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to encode the animation: {message}")


def _write_gif(
    source: Path,
    output: Path,
    *,
    fps: int = 15,
    width: int = 640,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the GIF preview")
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
    )
    completed = subprocess.run(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            filter_graph,
            "-loop",
            "0",
            str(output),
        ),
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to encode the GIF preview: {message}")


def render_crazyflow_bootstrap_animation(
    output: Path,
    *,
    poster_output: Path | None = None,
    gif_output: Path | None = None,
    config: CrazyflowAnimationConfig | None = None,
) -> dict[str, Any]:
    """Run the canonical diagnostic and render its exact telemetry to H.264."""

    config = CrazyflowAnimationConfig() if config is None else config
    if output.suffix.lower() != ".mp4":
        raise ValueError("animation output must have an .mp4 suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    if poster_output is not None:
        poster_output.parent.mkdir(parents=True, exist_ok=True)
    run = run_crazyflow_bootstrap_trial()
    trace = run.trace
    moments = _storyboard(trace, config)
    encoder = _start_encoder(
        output,
        width=config.width,
        height=config.height,
        frames_per_second=config.frames_per_second,
    )
    render_plant = CrazyflowPlant(CrazyflowPlantConfig())
    last_frame: np.ndarray | None = None
    try:
        for moment in moments:
            if moment.phase == "evidence":
                timestamps = trace.evidence_timestamps_s
                states = trace.evidence_states
                commands = trace.evidence_applied_motor_commands
            else:
                timestamps = trace.recovery_timestamps_s
                states = trace.recovery_states
                commands = trace.recovery_applied_motor_commands
            state, command, index = _interpolate_sample(
                timestamps,
                states,
                commands,
                moment.simulation_time_s,
            )
            trail = states[: index + 1, 0:3]
            rgb = _render_plant_frame(
                render_plant,
                state=state,
                command=command,
                trail=trail,
                width=config.width,
                height=config.height,
                recovery=moment.phase == "recovery",
            )
            frame = _draw_overlay(
                rgb,
                moment=moment,
                state=state,
                command=command,
                sample_index=index,
                trace=trace,
                report=run.report,
            )
            if encoder.stdin is None:
                raise RuntimeError("ffmpeg video input pipe disappeared")
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
            last_frame = frame
        _finish_encoder(encoder)
    except Exception:
        encoder.kill()
        encoder.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        render_plant.close()
    if poster_output is not None and last_frame is not None:
        from PIL import Image

        Image.fromarray(last_frame).save(poster_output)
    if gif_output is not None:
        _write_gif(output, gif_output)
    summary = {
        "output": str(output),
        "poster_output": None if poster_output is None else str(poster_output),
        "gif_output": None if gif_output is None else str(gif_output),
        "frame_count": len(moments),
        "frames_per_second": config.frames_per_second,
        "duration_s": len(moments) / config.frames_per_second,
        "source_artifact_type": run.report["artifact_type"],
        "gate_passed": run.report["observations"]["gate_passed"],
    }
    return summary


def render_crazyflow_throw_trace(
    trace: CrazyflowThrowTrace | CrazyflowStudyTrace,
    output: Path,
    *,
    poster_output: Path | None = None,
    gif_output: Path | None = None,
    gif_fps: int = 15,
    gif_width: int = 640,
    config: CrazyflowAnimationConfig | None = None,
    throw_only: bool = False,
) -> dict[str, Any]:
    """Render exact trace time to H.264, without a storyboard pause or reset.

    Takes any trace shaped like :class:`CrazyflowThrowTrace` (the standalone
    throw diagnostic) or
    :class:`~glassbox.integrations.crazyflow_throw_study.CrazyflowStudyTrace`
    (one study arm on one case) — the arm need not have certified, or even
    validated, a control belief.  Nothing here reads a report: every number on
    screen is read from ``trace`` itself.

    ``gif_fps``/``gif_width`` only take effect together with ``gif_output``;
    lower them for a busier scene whose default-settings preview would land
    above a size budget.
    """

    config = CrazyflowAnimationConfig() if config is None else config
    if output.suffix.lower() != ".mp4":
        raise ValueError("animation output must have an .mp4 suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    if poster_output is not None:
        poster_output.parent.mkdir(parents=True, exist_ok=True)
    moments = _throw_storyboard(trace, config, throw_only=throw_only)
    encoder = _start_encoder(
        output,
        width=config.width,
        height=config.height,
        frames_per_second=config.frames_per_second,
    )
    render_plant = CrazyflowPlant(CrazyflowPlantConfig())
    last_frame: np.ndarray | None = None
    try:
        for moment in moments:
            state, command, index = _interpolate_sample(
                trace.timestamps_s,
                trace.states,
                trace.applied_motor_commands,
                moment.simulation_time_s,
            )
            trail_start = max(0, index - 100)
            trail = trace.states[trail_start : index + 1, 0:3]
            rgb = _render_plant_frame(
                render_plant,
                state=state,
                command=command,
                trail=trail,
                width=config.width,
                height=config.height,
                recovery=True,
            )
            frame = _draw_throw_overlay(
                rgb,
                moment=moment,
                state=state,
                command=command,
                sample_index=index,
                trace=trace,
                throw_only=throw_only,
            )
            if encoder.stdin is None:
                raise RuntimeError("ffmpeg video input pipe disappeared")
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
            last_frame = frame
        _finish_encoder(encoder)
    except Exception:
        encoder.kill()
        encoder.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        render_plant.close()
    if poster_output is not None and last_frame is not None:
        from PIL import Image

        Image.fromarray(last_frame).save(poster_output)
    if gif_output is not None:
        _write_gif(output, gif_output, fps=gif_fps, width=gif_width)
    return {
        "output": str(output),
        "poster_output": None if poster_output is None else str(poster_output),
        "gif_output": None if gif_output is None else str(gif_output),
        "frame_count": len(moments),
        "frames_per_second": config.frames_per_second,
        "duration_s": len(moments) / config.frames_per_second,
        "arm": getattr(trace, "arm", "certified"),
        "case_name": getattr(trace, "case_name", "canonical"),
        "throw_only": throw_only,
        "real_time_playback": True,
    }


#: The fixed simulation times a contact sheet samples, in a 2x4 grid.
CONTACT_SHEET_TIMES_S: tuple[float, ...] = (0.0, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0, 10.0)


def render_crazyflow_throw_contact_sheet(
    trace: CrazyflowThrowTrace | CrazyflowStudyTrace,
    output: Path,
    *,
    config: CrazyflowAnimationConfig | None = None,
    times_s: Sequence[float] = CONTACT_SHEET_TIMES_S,
) -> dict[str, Any]:
    """Render a fixed 2x4 grid of frames, each stamped with its own time.

    Every tile is one exact rendered frame at the requested simulation time,
    clipped to the trace's own span; the label stamped on a tile is the time
    the trace actually had there, not the requested one, so a request past the
    end of a shorter trace reads honestly rather than silently repeating the
    last frame's own time.
    """

    from PIL import Image, ImageDraw

    config = CrazyflowAnimationConfig() if config is None else config
    if output.suffix.lower() != ".png":
        raise ValueError("contact sheet output must have a .png suffix")
    if len(times_s) != 8:
        raise ValueError("the contact sheet lays out exactly eight tiles, 2x4")
    output.parent.mkdir(parents=True, exist_ok=True)
    tile_width = (config.width // 4) - (config.width // 4) % 2
    tile_height = (config.height // 2) - (config.height // 2) % 2
    render_plant = CrazyflowPlant(CrazyflowPlantConfig())
    sheet = Image.new("RGB", (tile_width * 4, tile_height * 2), color=_NAVY)
    try:
        for tile_index, requested_time_s in enumerate(times_s):
            state, command, index = _interpolate_sample(
                trace.timestamps_s,
                trace.states,
                trace.applied_motor_commands,
                float(requested_time_s),
            )
            trail_start = max(0, index - 100)
            trail = trace.states[trail_start : index + 1, 0:3]
            rgb = _render_plant_frame(
                render_plant,
                state=state,
                command=command,
                trail=trail,
                width=tile_width,
                height=tile_height,
                recovery=True,
            )
            tile = Image.fromarray(rgb)
            draw = ImageDraw.Draw(tile, "RGBA")
            scale = tile_width / 320.0
            font = _load_font(max(12, round(20 * scale)), bold=True, mono=True)
            band_top = tile_height - round(28 * scale)
            draw.rectangle((0, band_top, tile_width, tile_height), fill=(*_NAVY, 210))
            draw.text(
                (round(8 * scale), band_top + round(6 * scale)),
                f"t = {float(trace.timestamps_s[index]):5.2f} s",
                font=font,
                fill=(*_WHITE, 255),
            )
            row, column = divmod(tile_index, 4)
            sheet.paste(tile, (column * tile_width, row * tile_height))
        sheet.save(output)
    finally:
        render_plant.close()
    return {
        "output": str(output),
        "tile_count": len(times_s),
        "tile_width": tile_width,
        "tile_height": tile_height,
        "times_s": [float(time_s) for time_s in times_s],
        "arm": getattr(trace, "arm", "certified"),
        "case_name": getattr(trace, "case_name", "canonical"),
    }


def render_crazyflow_throw_animation(
    output: Path,
    *,
    poster_output: Path | None = None,
    gif_output: Path | None = None,
    config: CrazyflowAnimationConfig | None = None,
) -> dict[str, Any]:
    """Render the uninterrupted one-second throw and online recovery.

    A thin wrapper: it runs the default trial — the frozen-snapshot cascade on
    the canonical release, unchanged — and hands its trace to
    :func:`render_crazyflow_throw_trace`.
    """

    run = run_crazyflow_throw_trial()
    return render_crazyflow_throw_trace(
        run.trace,
        output,
        poster_output=poster_output,
        gif_output=gif_output,
        config=config,
    )


def render_crazyflow_unpowered_throw_animation(
    output: Path,
    *,
    poster_output: Path | None = None,
    gif_output: Path | None = None,
    config: CrazyflowAnimationConfig | None = None,
) -> dict[str, Any]:
    """Render only the exact unpowered first second of the same demo trace."""

    run = run_crazyflow_throw_trial()
    return render_crazyflow_throw_trace(
        run.trace,
        output,
        poster_output=poster_output,
        gif_output=gif_output,
        config=config,
        throw_only=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    # Set before any lazy `import crazyflow` reaches its own SciPy-array-API guard.
    os.environ.setdefault("SCIPY_ARRAY_API", "1")
    parser = argparse.ArgumentParser(
        description="Render the no-prior Crazyflow bootstrap diagnostic."
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("artifacts/crazyflow_bootstrap/no-prior-bootstrap.mp4"),
    )
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)
    summary = render_crazyflow_bootstrap_animation(
        args.output,
        poster_output=args.poster,
        gif_output=args.gif,
        config=CrazyflowAnimationConfig(
            width=args.width,
            height=args.height,
            frames_per_second=args.fps,
        ),
    )
    print(
        f"wrote {summary['output']} "
        f"({summary['duration_s']:.1f} s, {summary['frame_count']} frames)"
    )


def throw_main(argv: Sequence[str] | None = None) -> None:
    # Set before any lazy `import crazyflow` reaches its own SciPy-array-API guard.
    os.environ.setdefault("SCIPY_ARRAY_API", "1")
    parser = argparse.ArgumentParser(
        description="Render the real-time unpowered-throw Crazyflow diagnostic."
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("artifacts/crazyflow_throw/continuous-throw.mp4"),
    )
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--gif", type=Path)
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=15,
        help="frame rate of the GIF preview, independent of --fps (default: %(default)s)",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=640,
        help="pixel width of the GIF preview (default: %(default)s)",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        help="also render a 2x4 grid of stamped frames at fixed simulation times",
    )
    parser.add_argument(
        "--throw-only-output",
        type=Path,
        help="also render the exact unpowered first second",
    )
    parser.add_argument("--throw-only-poster", type=Path)
    parser.add_argument("--throw-only-gif", type=Path)
    parser.add_argument(
        "--control-model",
        choices=STUDY_CONTROL_MODELS,
        default=None,
        help=(
            "render one glassbox crazyflow throw-study arm instead of the "
            "standalone throw diagnostic (default: the standalone diagnostic, "
            "which is the frozen-snapshot cascade on the canonical release)"
        ),
    )
    parser.add_argument(
        "--case",
        default="canonical",
        choices=[case.name for case in CRAZYFLOW_THROW_STUDY_CASES],
        help=(
            "throw-study case to render when --control-model is given "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)
    config = CrazyflowAnimationConfig(
        width=args.width,
        height=args.height,
        frames_per_second=args.fps,
    )
    if args.control_model is None:
        trace: CrazyflowThrowTrace | CrazyflowStudyTrace = (
            run_crazyflow_throw_trial().trace
        )
    else:
        case = next(
            case for case in CRAZYFLOW_THROW_STUDY_CASES if case.name == args.case
        )
        trace = run_throw_study_render_trial(case, args.control_model)
    summary = render_crazyflow_throw_trace(
        trace,
        args.output,
        poster_output=args.poster,
        gif_output=args.gif,
        gif_fps=args.gif_fps,
        gif_width=args.gif_width,
        config=config,
    )
    print(
        f"wrote {summary['output']} "
        f"({summary['duration_s']:.1f} s, {summary['frame_count']} frames)"
    )
    if args.throw_only_output is not None:
        throw_summary = render_crazyflow_throw_trace(
            trace,
            args.throw_only_output,
            poster_output=args.throw_only_poster,
            gif_output=args.throw_only_gif,
            gif_fps=args.gif_fps,
            gif_width=args.gif_width,
            config=config,
            throw_only=True,
        )
        print(
            f"wrote {throw_summary['output']} "
            f"({throw_summary['duration_s']:.1f} s, "
            f"{throw_summary['frame_count']} frames)"
        )
    if args.contact_sheet is not None:
        contact_summary = render_crazyflow_throw_contact_sheet(
            trace,
            args.contact_sheet,
            config=config,
        )
        print(
            f"wrote {contact_summary['output']} ({contact_summary['tile_count']} tiles)"
        )


if __name__ == "__main__":
    main()
