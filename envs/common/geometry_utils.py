"""Geometry utilities for transformations."""

import numpy as np
from numpy.typing import NDArray


def to_4x4(R: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert rotation matrix and translation vector to 4x4 transformation matrix.

    Args:
        R: 3x3 rotation matrix
        t: 3D translation vector

    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

