"""YAM pick-and-place block task environment."""

# Mujoco typing preamble: incomplete types from mujoco.
# pyright: reportUnknownMemberType=none
# pyright: reportAttributeAccessIssue=none

from typing import Any, Literal, Union

import mujoco
import numpy as np

from envs.managers import (
    ActionManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from envs.mj_env import MJSceneManager, MJState
from envs.randomization import (
    MjFreeJointRandomizer,
    RandomizedMJSceneManager,
    RotationDistribution,
)

from ..core.env import YAMTableTopEnv

DEFAULT_PLACE_POS = np.array([0.1, -0.1, 0.47], dtype=np.float64)
BLOCK_SIZE = [0.025, 0.025, 0.025]


def add_block_to_spec(spec: mujoco.MjSpec, table_pos: tuple[float, float, float], table_size: tuple[float, float, float], block_pos: tuple[float, float, float] | None = None) -> None:
    """Add a block object to a MuJoCo spec."""
    if block_pos is None:
        table_top_z = table_pos[2] + table_size[2] / 2
        block_pos = (0.2, 0.1, table_top_z + BLOCK_SIZE[2])

    block = spec.worldbody.add_body()
    block.name = "block"
    block.pos = list(block_pos)

    block_geom = block.add_geom()
    block_geom.name = "block_geom"
    block_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    block_geom.size = [s / 2.0 for s in BLOCK_SIZE]
    block_geom.rgba = [0.8, 0.2, 0.2, 1]
    block_geom.mass = 0.05
    block_geom.friction = [2.0, 0.1, 0.001]

    block_joint = block.add_freejoint()
    block_joint.name = "block_freejoint"

    block_frame = block.add_site()
    block_frame.name = "block_frame"
    block_frame.type = mujoco.mjtGeom.mjGEOM_SPHERE
    block_frame.size = [0.01, 0.01, 0.01]
    block_frame.rgba = [1, 1, 0, 1]
    block_frame.pos = [0, 0, 0]


def add_goal_marker_to_spec(spec: mujoco.MjSpec, goal_location: np.ndarray) -> None:
    """Add a goal marker to a MuJoCo spec."""
    goal_body = spec.worldbody.add_body()
    goal_body.name = "goal_marker_body"
    goal_body.pos = list(goal_location)

    goal_site = goal_body.add_site()
    goal_site.name = "goal_marker"
    goal_site.type = mujoco.mjtGeom.mjGEOM_SPHERE
    goal_site.size = [0.01, 0.01, 0.01]
    goal_site.rgba = [0, 1, 0, 0.3]
    goal_site.pos = [0, 0, 0]


class YAMPickAndPlaceBlockEnv(YAMTableTopEnv):
    """YAM pick-and-place block task environment."""

    def __init__(
        self,
        table_size: tuple[float, float, float] = (0.6, 0.4, 0.02),
        table_pos: tuple[float, float, float] = (0.0, 0.0, 0.4),
        table_friction: tuple[float, float, float] = (2.0, 0.1, 0.001),
        timestep: float = 0.002,
        integrator: mujoco.mjtIntegrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST,  # type: ignore[assignment]
        time_limit: float | None = None,
        max_steps: int | None = None,
        block_pos: tuple[float, float, float] | None = None,
        goal_location: np.ndarray | None = None,
        camera_type: Literal["fixed", "tracking"] = "tracking",
        observation_manager: ObservationManager[MJState, Any] | None = None,
        action_manager: ActionManager[np.ndarray, np.ndarray] | None = None,
        reward_manager: RewardManager[MJState] | None = None,
        termination_manager: TerminationManager[MJState] | None = None,
        randomize_block: bool = False,
        block_x_range: tuple[float, float] | None = (0.1, 0.3),
        block_y_range: tuple[float, float] | None = (-0.1, 0.1),
        block_z_range: tuple[float, float] | None = None,
        block_rotation_distribution: RotationDistribution | None = RotationDistribution.RANDOM_Z,
        fps: float | None = None,
        enable_rendering: bool = False,
        render_width: int = 1280,
        render_height: int = 720,
        action_mode: Literal["joints", "pose"] = "joints",
    ):
        """Initialize the YAM pick-and-place block environment.

        Args:
            table_size: (L, W, H) dimensions of the table top.
            table_pos: (x, y, z) position of the table center.
            table_friction: (sliding, torsional, rolling) friction parameters.
            timestep: Simulation timestep in seconds.
            integrator: MuJoCo integrator type.
            time_limit: Maximum episode time in seconds (mutually exclusive with max_steps).
            max_steps: Maximum episode steps (mutually exclusive with time_limit).
            block_pos: Optional position for the block. If None, uses default position on table.
            goal_location: Optional goal position [x, y, z] to add visual marker.
            camera_type: Type of camera - "fixed" or "tracking".
            observation_manager: Custom observation manager. If None, uses identity.
            action_manager: Custom action manager. If None, uses identity.
            reward_manager: Custom reward manager. If None, uses zero reward.
            termination_manager: Custom termination manager. If None, uses timeout only.
            randomize_block: If True, randomize block position and rotation on each reset.
            block_x_range: X position range for randomization (min, max). Only used if randomize_block=True.
            block_y_range: Y position range for randomization (min, max). Only used if randomize_block=True.
            block_z_range: Z position range for randomization (min, max). If None and randomize_block=True,
                uses table top height + block half-height. Only used if randomize_block=True.
            block_rotation_distribution: Rotation distribution for randomization. Only used if randomize_block=True.
            fps: Target frequency for actions in Hz. If None, actions are executed at simulation timestep frequency.
                When set, actions are provided at this frequency and interpolated over multiple simulation steps.
            action_mode: Action mode - "joints" for joint angle actions (7 dims) or "pose" for end effector pose actions (10 dims: 3D pos + 6D rot + gripper).
        """
        # Store task-specific parameters for use in overridden methods
        self._block_pos = block_pos
        self._goal_location = goal_location
        self._randomize_block = randomize_block
        self._block_x_range = block_x_range
        self._block_y_range = block_y_range
        self._block_z_range = block_z_range
        self._block_rotation_distribution = block_rotation_distribution
        self._table_size = table_size
        self._table_pos = table_pos

        # Call parent initialization with all common parameters
        super().__init__(
            table_size=table_size,
            table_pos=table_pos,
            table_friction=table_friction,
            timestep=timestep,
            integrator=integrator,
            time_limit=time_limit,
            max_steps=max_steps,
            camera_type=camera_type,
            observation_manager=observation_manager,
            action_manager=action_manager,
            reward_manager=reward_manager,
            termination_manager=termination_manager,
            fps=fps,
            enable_rendering=enable_rendering,
            render_width=render_width,
            render_height=render_height,
            action_mode=action_mode,
        )

        # Set task-specific attributes
        self.obj_of_interest = ["block"]

    def _customize_scene_spec(
        self,
        spec: mujoco.MjSpec,
        table_size: tuple[float, float, float],
        table_pos: tuple[float, float, float],
    ) -> mujoco.MjSpec:
        """Add block and goal marker to the scene spec."""
        add_block_to_spec(spec, table_pos, table_size, block_pos=self._block_pos)
        if self._goal_location is not None:
            add_goal_marker_to_spec(spec, self._goal_location)
        return spec

    def _create_scene_manager(self, model: mujoco.MjModel) -> Union[MJSceneManager, RandomizedMJSceneManager]:  # type: ignore[override]
        """Create scene manager with block randomization if enabled."""
        base_scene_manager = MJSceneManager(model=model)

        if self._randomize_block:
            table_top_z = self._table_pos[2] + self._table_size[2] / 2
            block_z_range = self._block_z_range
            if block_z_range is None:
                block_z_range = (table_top_z + BLOCK_SIZE[2], table_top_z + BLOCK_SIZE[2])

            randomizers = [
                MjFreeJointRandomizer(
                    body_name="block",
                    x_range=self._block_x_range,
                    y_range=self._block_y_range,
                    z_range=block_z_range,
                    rotation_distribution=self._block_rotation_distribution,
                )
            ]
            return RandomizedMJSceneManager(base_scene_manager, randomizers)
        else:
            return base_scene_manager

