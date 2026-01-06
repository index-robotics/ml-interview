"""Free joint position and rotation randomizer."""

from enum import Enum
from typing import cast, final

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class RotationDistribution(str, Enum):
    """Distribution types for rotation randomization."""

    RANDOM_Z = "random_z"  # Random rotation around Z-axis only
    RANDOM_SO3 = "random_so3"  # Uniformly random rotation in SO(3)


@final
class MjFreeJointRandomizer:
    """Randomizer for free joint position and orientation.

    This randomizer samples the position and/or orientation of a body with a
    freejoint. Position is sampled uniformly within specified ranges for each
    axis. Rotation can be uniformly random around the Z-axis or uniformly
    random in SO(3).

    Example:
        >>> # Randomize block position on table surface
        >>> randomizer = MjFreeJointRandomizer(
        ...     body_name="block",
        ...     x_range=(-0.2, 0.2),
        ...     y_range=(-0.1, 0.1),
        ...     z_range=None,  # Keep Z fixed
        ...     rotation_distribution=RotationDistribution.RANDOM_Z,
        ... )
    """

    def __init__(
        self,
        body_name: str,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
        rotation_distribution: RotationDistribution | None = None,
    ):
        """Initialize free joint randomizer.

        Args:
            body_name: Name of the body with freejoint to randomize
            x_range: Optional (min, max) range for X position
            y_range: Optional (min, max) range for Y position
            z_range: Optional (min, max) range for Z position
            rotation_distribution: Optional rotation distribution type
        """
        self.body_name = body_name
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.rotation_distribution = rotation_distribution

        # Cache body and joint IDs (set during first reset)
        self._body_id: int | None = None
        self._joint_id: int | None = None
        self._validated = False

    def _validate_and_cache(self, model: mujoco.MjModel) -> None:
        """Validate that the body has a freejoint and cache IDs.

        Args:
            model: MuJoCo model

        Raises:
            ValueError: If body not found or doesn't have a freejoint
        """
        if self._validated:
            return

        # Find body ID
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if body_id == -1:
            raise ValueError(f"Body '{self.body_name}' not found in model")

        # Check if body has exactly one joint and it's a freejoint
        # Bodies with freejoints have jntadr pointing to their first joint
        body_jntnum = cast(int, model.body_jntnum[body_id])
        if body_jntnum == 0:
            raise ValueError(
                f"Body '{self.body_name}' has no joints. "
                + "Free joint randomization requires a freejoint."
            )

        # Get the first joint of this body
        body_jntadr = cast(int, model.body_jntadr[body_id])
        joint_type = cast(int, model.jnt_type[body_jntadr])

        # Check if it's a freejoint (mjtJoint.mjJNT_FREE = 0)
        if joint_type != mujoco.mjtJoint.mjJNT_FREE:
            joint_type_name = mujoco.mjtJoint(joint_type).name
            raise ValueError(
                f"Body '{self.body_name}' does not have a freejoint. "
                + f"Found joint type: {joint_type_name}. "
                + "Free joint randomization requires a freejoint."
            )

        self._body_id = body_id
        self._joint_id = body_jntadr
        self._validated = True

    def reset(
        self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator
    ) -> None:
        """Apply randomization to free joint position and orientation.

        Args:
            model: MuJoCo model
            data: MuJoCo data to modify
            rng: Random number generator
        """
        # Validate and cache IDs on first call
        self._validate_and_cache(model)

        assert self._joint_id is not None

        # Get qpos address for this joint
        # Freejoint has 7 DOFs: 3 for position (xyz), 4 for quaternion (wxyz)
        qpos_adr = cast(int, model.jnt_qposadr[self._joint_id])

        # Randomize position (first 3 elements)
        if self.x_range is not None:
            data.qpos[qpos_adr + 0] = rng.uniform(*self.x_range)
        if self.y_range is not None:
            data.qpos[qpos_adr + 1] = rng.uniform(*self.y_range)
        if self.z_range is not None:
            data.qpos[qpos_adr + 2] = rng.uniform(*self.z_range)

        # Randomize orientation (quaternion at indices 3-6, stored as wxyz)
        if self.rotation_distribution is not None:
            if self.rotation_distribution == RotationDistribution.RANDOM_Z:
                # Random rotation around Z-axis
                angle = rng.uniform(0, 2 * np.pi)
                rotation = Rotation.from_euler("z", angle)
            elif self.rotation_distribution == RotationDistribution.RANDOM_SO3:
                # Uniformly random rotation in SO(3)
                rotation = Rotation.random(random_state=rng)
            else:
                raise ValueError(
                    f"Unknown rotation distribution: {self.rotation_distribution}"
                )

            # Convert to quaternion (scipy returns xyzw, MuJoCo uses wxyz)
            quat_xyzw = rotation.as_quat()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz
