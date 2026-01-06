"""YAM robot definition for tabletop manipulation."""

import numpy as np


class YAMRobot:
    """YAM robot configuration and properties.

    This class provides robot-specific information similar to how MimicLabs
    defines robot models (e.g., MountedPanda).
    """

    def __init__(self, idn: int = 0):
        """Initialize YAM robot configuration.

        Args:
            idn: Robot instance identifier (for multi-robot scenarios).
        """
        self.idn = idn
        self.robot_name = "yam_arm"

    @property
    def default_mount(self) -> str:
        """Default mount type for the robot."""
        return "TableMount"

    @property
    def default_gripper(self) -> str:
        """Default gripper type."""
        return "YAMGripper"

    @property
    def init_qpos(self) -> np.ndarray:
        """Initial joint positions for the robot.

        Returns:
            Array of initial joint positions (7 DOF for YAM arm).
        """
        # Default to a neutral/home position
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def base_xpos_offset(self) -> dict[str, tuple[float, float, float]]:
        """Base position offsets for different arena configurations.

        Returns:
            Dictionary mapping arena types to (x, y, z) offset tuples.
        """
        return {
            "table": (0.0, 0.0, 0.0),  # Centered on table
            "empty": (0.0, 0.0, 0.0),
        }

    @property
    def top_offset(self) -> np.ndarray:
        """Offset from base to top of robot workspace.

        Returns:
            (x, y, z) offset array.
        """
        return np.array([0.0, 0.0, 0.5])

    @property
    def horizontal_radius(self) -> float:
        """Horizontal reach radius of the robot in meters."""
        return 0.4  # Approximate reach of YAM arm

    @property
    def arm_type(self) -> str:
        """Type of arm configuration."""
        return "single"

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        return 7

    @property
    def gripper_dof(self) -> int:
        """Number of gripper degrees of freedom."""
        return 0  # YAM uses position control for gripper, not separate DOF

    @property
    def control_dof(self) -> int:
        """Total control degrees of freedom (arm + gripper)."""
        return self.n_dof + self.gripper_dof

