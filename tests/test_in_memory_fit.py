"""The structured benchmark fitter preserves semantics across storage forms."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from glassbox.core.data import (
    resolve_trajectory_sources,
    save_trajectory_npz,
    trajectory_content_digest,
)
from glassbox.fitting import FitSpec, Holdout, fit, fit_report_digest


@pytest.mark.parametrize("family", ["quadrotor", "fixedwing"])
def test_fit_storage_forms_preserve_windows_parameters_and_evidence(
    family, request, tmp_path, monkeypatch
):
    generate = request.getfixturevalue(f"{family}_flight")
    paths = [tmp_path / f"flight-{i}.npz" for i in range(3)]
    flights = [
        replace(
            generate(i),
            labels={"source_group": f"session-{i}"},
            provenance={"path": str(path), "recording": f"flight-{i}"},
        )
        for i, path in enumerate(paths)
    ]
    for flight, path in zip(flights, paths):
        save_trajectory_npz(flight, path)
    spec = FitSpec(steps=2, horizon_steps=5, evaluation_horizons_s=(0.1,))

    file_outcome = fit(paths, spec)
    mixed_outcome = fit([flights[0], paths[1], flights[2]], spec)

    def unexpected_read(*args, **kwargs):
        pytest.fail("an in-memory fit must not read trajectory files")

    with monkeypatch.context() as patch:
        patch.setattr("glassbox.core.data.load_trajectory_npz", unexpected_read)
        memory_outcome = fit(flights, spec)

    expected = fit_report_digest(file_outcome.report)
    for outcome in (mixed_outcome, memory_outcome):
        # The complete report includes selected windows, fitted parameters,
        # reserved errors and information. Only wall-clock timing is excluded.
        assert fit_report_digest(outcome.report) == expected
        assert outcome.belief.provenance == file_outcome.belief.provenance
        np.testing.assert_array_equal(
            outcome.belief.information.precision,
            file_outcome.belief.information.precision,
        )
    split = memory_outcome.report["split"]
    assert split["training_source_groups"] == ["session-0", "session-1"]
    assert split["validation_source_groups"] == ["session-2"]
    assert split["validation_role"] == "forecast_error_calibration"
    assert memory_outcome.belief.provenance["data_identity"][
        "forecast_error_calibration"
    ] == [trajectory_content_digest(flights[2])]


@pytest.mark.parametrize("family", ["quadrotor", "fixedwing"])
def test_temporal_memory_fit_retains_real_command_prefix(family, request, tmp_path):
    generate = request.getfixturevalue(f"{family}_flight")
    path = tmp_path / "flight.npz"
    flight = replace(generate(0), provenance={"path": str(path)})
    save_trajectory_npz(flight, path)
    spec = FitSpec(
        holdout=Holdout.temporal(),
        steps=1,
        horizon_steps=3,
        evaluation_horizons_s=(0.04,),
    )
    memory = fit([flight], spec)
    stored = fit([path], spec)
    assert fit_report_digest(memory.report) == fit_report_digest(stored.report)
    plan = spec.holdout.plan([flight])
    reserved = plan.validation[0].trajectory
    assert reserved.control_prefix is not None
    assert len(reserved.control_prefix) == len(plan.training[0].controls)
    identity = memory.belief.provenance["data_identity"]
    assert identity["forecast_error_calibration"] == [
        trajectory_content_digest(reserved)
    ]
    assert identity["training"] != identity["forecast_error_calibration"]
    assert memory.report["split"]["independent_source_group_holdout"] is False


@pytest.mark.parametrize("family", ["quadrotor", "fixedwing"])
@pytest.mark.parametrize("storage", ["memory", "files", "mixed"])
@pytest.mark.parametrize("mismatch", ["configuration", "timing", "channel"])
def test_mixed_fit_rejects_incompatible_contracts_before_fitting(
    family, storage, mismatch, request, tmp_path
):
    generate = request.getfixturevalue(f"{family}_flight")
    first = generate(0)
    second = generate(1)
    expected = "inconsistent dataset trajectory_spec"
    if mismatch == "configuration":
        second = replace(
            second,
            spec=replace(
                second.spec,
                vehicle=replace(second.spec.vehicle, configuration_id="other-revision"),
            ),
        )
    elif mismatch == "timing":
        second = replace(second, time_s=second.time_s * 2.0)
        expected = "inconsistent dataset sample_rate_hz"
    else:
        channels = second.spec.channels
        second = replace(
            second,
            spec=replace(
                second.spec,
                channels=(replace(channels[0], maximum=0.9), *channels[1:]),
            ),
        )
    sources = [first, second]
    for index, flight in enumerate(sources):
        if storage == "files" or (storage == "mixed" and index == 1):
            path = tmp_path / f"{index}.npz"
            save_trajectory_npz(flight, path)
            sources[index] = path
    with pytest.raises(ValueError, match=expected):
        fit(sources, FitSpec(steps=1))


def test_source_resolution_keeps_memory_names_distinct_from_files(
    quadrotor_flight, tmp_path
):
    path = tmp_path / "flight.npz"
    flight = replace(
        quadrotor_flight(0), provenance={"path": f"{tmp_path}/./flight.npz"}
    )
    save_trajectory_npz(flight, path)
    labels, flights = resolve_trajectory_sources([flight, path, flight])
    assert labels[1] == str(path)
    assert len(set(labels)) == 3
    assert len({str(Path(label)) for label in labels}) == 3
    assert flights[0] is flight
    assert flights[2] is flight
    assert len({trajectory_content_digest(item) for item in flights}) == 1


def test_fit_rejects_empty_or_nontrajectory_inputs():
    with pytest.raises(ValueError, match="at least one trajectory"):
        fit([])
    with pytest.raises(TypeError, match="source 0 must be a Trajectory"):
        fit([np.zeros((5, 13))])
