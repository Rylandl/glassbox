"""Shared PX4-to-Glassbox rigid-body frame conversions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

PX4_NED_TO_NWU_SIGNS = np.asarray([1.0, -1.0, -1.0])
PX4_FRD_TO_FLU_SIGNS = np.asarray([1.0, -1.0, -1.0])
PX4_NED_FRD_TO_NWU_FLU_QUATERNION_SIGNS = np.asarray([1.0, 1.0, -1.0, -1.0])


def _convert(
    values: npt.ArrayLike,
    signs: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != len(signs):
        raise ValueError(f"{name} must have final dimension {len(signs)}")
    return array * signs


def ned_to_nwu(values: npt.ArrayLike) -> np.ndarray:
    """Convert world-frame vectors from PX4 NED to Glassbox NWU."""

    return _convert(values, PX4_NED_TO_NWU_SIGNS, name="NED vector")


def frd_to_flu(values: npt.ArrayLike) -> np.ndarray:
    """Convert body-frame vectors from PX4 FRD to Glassbox FLU."""

    return _convert(values, PX4_FRD_TO_FLU_SIGNS, name="FRD vector")


def ned_frd_quaternion_to_nwu_flu(values: npt.ArrayLike) -> np.ndarray:
    """Convert WXYZ attitude quaternions from NED/FRD to NWU/FLU."""

    return _convert(
        values,
        PX4_NED_FRD_TO_NWU_FLU_QUATERNION_SIGNS,
        name="NED/FRD quaternion",
    )
