"""Interpolation utilities."""

import numpy as np
from numpy.typing import NDArray


def linear_interpolate(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    num_steps: int,
) -> list[NDArray[np.float64]]:
    """Linearly interpolate between start and end values.

    Args:
        start: Starting values
        end: Ending values
        num_steps: Number of interpolation steps (including endpoints)

    Returns:
        List of interpolated arrays, including start and end
    """
    if num_steps < 2:
        return [end.copy()]

    alphas = np.linspace(0.0, 1.0, num_steps)
    interpolated = []
    for alpha in alphas:
        interpolated.append((1 - alpha) * start + alpha * end)
    return interpolated

