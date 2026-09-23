"""Read-only motion forecasts on complete, previously unseen recordings."""

from __future__ import annotations

import numpy as np

from glassbox.learner import LearnedDynamics, _contract, _recording_content
from glassbox.recordings import SequenceCollection


def evaluate(model: LearnedDynamics, recordings: SequenceCollection) -> dict:
    """Measure each complete causal-prefix/horizon window without fitting.

    Every reported array is [horizon step, observation channel], in the order
    declared by the model. Errors are in each channel's own units. Coverage
    counts observations inside the saved envelope; it is not an acceptance rule.
    Exact recording identities/content are checked against both training and
    development. Different segmentation is not proof of recording independence.
    """
    if not isinstance(model, LearnedDynamics):
        raise TypeError("forecast evaluation requires a LearnedDynamics model")
    if _contract(recordings) != model.contract:
        raise ValueError("evaluation configuration, channels or sample interval differ")
    content = _recording_content(recordings)
    if set(content) & set(model._seen) or set(content.values()) & set(
        model._seen.values()
    ):
        raise ValueError(
            "evaluation requires recording identities and content outside fitting"
        )
    horizon = model.horizon_steps
    shape = (horizon, len(model.contract["state_channels"]))
    calibration = model.report.get("envelope")
    if calibration is not None and calibration.get("available") is False:
        calibration = None
    envelope = model.envelope() if calibration is not None else None

    def empty():
        return dict(
            windows=0,
            squared=np.zeros(shape),
            hold=np.zeros(shape),
            covered=np.zeros(shape),
        )

    totals = empty()
    by_recording = {name: empty() for name in sorted(content)}
    for segment in recordings.segments:
        origins = np.arange(1, len(segment.states) - horizon)
        if not len(origins):
            continue
        truth = np.stack([segment.states[t + 1 : t + horizon + 1] for t in origins])
        prediction = np.asarray(model._predict_origins(segment, origins, horizon))
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError(
                "model produced a nonfinite or misaligned evaluation forecast"
            )
        error = prediction - truth
        hold_error = segment.states[origins, None, :] - truth
        for target in (totals, by_recording[segment.recording_id]):
            target["windows"] += len(origins)
            target["squared"] += np.sum(error**2, axis=0)
            target["hold"] += np.sum(hold_error**2, axis=0)
            if envelope is not None:
                target["covered"] += np.sum(np.abs(error) <= envelope, axis=0)
    missing = [name for name, value in by_recording.items() if value["windows"] == 0]
    if missing:
        raise ValueError(
            f"recordings have no complete causal-prefix/horizon window: {missing}"
        )

    def metrics(value):
        count = value["windows"]
        return dict(
            windows=count,
            rmse=np.sqrt(value["squared"] / count).tolist(),
            hold_current_rmse=np.sqrt(value["hold"] / count).tolist(),
            coverage=(
                (value["covered"] / count).tolist() if envelope is not None else None
            ),
        )

    return dict(
        format="glassbox-motion-forecast-evaluation-v1",
        model_fingerprint=model.fingerprint(),
        contract=model.contract,
        history_steps=None,
        history_policy="complete_segment",
        horizon_steps=horizon,
        horizons_s=(np.arange(1, horizon + 1) * model.contract["dt_s"]).tolist(),
        nominal_coverage=(
            calibration["nominal_coverage"] if calibration is not None else None
        ),
        calibration_provenance=(
            {key: value for key, value in calibration.items() if key != "half_width"}
            if calibration is not None
            else None
        ),
        known_recording_count=len(model._seen),
        known_recording_reuse_detected=False,
        independence_check=(
            "recording IDs and exact content against retained identities; "
            "caller owns recording boundaries and independence"
        ),
        weighting="every_complete_window",
        aggregate=metrics(totals),
        per_recording={name: metrics(value) for name, value in by_recording.items()},
    )
