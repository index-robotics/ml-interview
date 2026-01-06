"""Robot utility functions for learning and manipulation."""

import numpy as np
from numpy.typing import NDArray


def rotation_6d_to_matrix(rot_6d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert 6D rotation representation to rotation matrix.

    The 6D representation uses the first two columns of the rotation matrix.
    The third column is recovered via cross product.

    Args:
        rot_6d: 6D rotation representation (6,) array

    Returns:
        3x3 rotation matrix
    """
    rot_6d = np.array(rot_6d, dtype=np.float64).reshape(6)
    col1 = rot_6d[:3]
    col2 = rot_6d[3:6]

    # Normalize columns
    col1 = col1 / np.linalg.norm(col1)
    col2 = col2 / np.linalg.norm(col2)

    # Recover third column via cross product
    col3 = np.cross(col1, col2)
    col3 = col3 / np.linalg.norm(col3)

    # Re-orthogonalize col2
    col2 = np.cross(col3, col1)
    col2 = col2 / np.linalg.norm(col2)

    R = np.column_stack([col1, col2, col3])
    return R


# Transformation matrix from gripper body frame to gripper site frame
# This is a constant transformation for the YAM robot gripper
T_gripper: NDArray[np.float64] = np.eye(4, dtype=np.float64)
# If there's a specific offset, it should be set here
# For now, assuming identity (gripper site is at gripper body origin)

