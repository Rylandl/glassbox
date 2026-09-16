"""Protocol and empirical selector checks independent of the fitted flight data."""

import runpy
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def scripts(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(folder))
    return folder


@pytest.mark.parametrize("version", [1, 2])
def test_usd_decoder_timestamps_fields_and_crc(tmp_path, scripts, version):
    decode = runpy.run_path(str(scripts / "prepare_crazyflie_reference.py"))["decode"]
    header = (
        b"\xbc"
        + struct.pack("<HHH", version, 1, 3)
        + b"fixedFrequency\0"
        + struct.pack("<H", 2)
        + b"gyro.x[f]\0motor.m1[H]\0"
    )
    clock = "<HI" if version == 1 else "<HQ"
    payload = header + b"".join(
        struct.pack(clock, 3, t) + struct.pack("<fH", value, motor)
        for t, value, motor in [(1000, 1.5, 100), (2000, -2.0, 200)]
    )
    path = tmp_path / "log"
    path.write_bytes(payload + struct.pack("<I", zlib.crc32(payload)))
    actual = decode(path)["fixedFrequency"]
    np.testing.assert_allclose(
        actual["timestamp_s"], np.array([1000, 2000]) / (1000 if version == 1 else 1e6)
    )
    np.testing.assert_array_equal(actual["gyro.x"], [1.5, -2])
    np.testing.assert_array_equal(actual["motor.m1"], [100, 200])
    path.write_bytes(path.read_bytes()[:-1] + b"\0")
    with pytest.raises(ValueError, match="CRC"):
        decode(path)


def test_local_selector_uses_reserved_residuals_and_query_features(scripts):
    choose = runpy.run_path(str(scripts / "experiment_forecast_representation.py"))[
        "local_choice"
    ]
    training = np.array([[-2.0], [2.0]])
    development = np.array([[-1.0], [1.0]])
    residual = np.array([[[[1.0]], [[9.0]]], [[[9.0]], [[1.0]]]])
    predictions = np.array([[[[10.0]], [[10.0]]], [[[20.0]], [[20.0]]]])
    mean, evidence = choose(
        training, development, residual, development, predictions, ((0,),), neighbors=1
    )
    np.testing.assert_array_equal(mean[:, 0, 0], [10, 20])
    np.testing.assert_array_equal(evidence["indices"], [[0], [1]])
