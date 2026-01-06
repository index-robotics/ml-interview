"""Core YAM tabletop manipulation environment (base environment without task-specific objects)."""

# Mujoco typing preamble: incomplete types from mujoco.
# pyright: reportUnknownMemberType=none
# pyright: reportAttributeAccessIssue=none

from typing import Any, Literal

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.mj_env import (
    MJManagerBasedEnv,
    MJSceneManager,
    MJState,
    TimeoutTerminationManager,
    mjData_from_mjState,
    raw_action_space,
    raw_observation_space,
)
from envs.managers import (
    ActionManager,
    IdentityActionManager,
    IdentityObservationManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from envs.common.camera_config import FixedCamera, TrackingCamera
from envs.common.interpolators import linear_interpolate
from envs.planning.trajectory import LinearJointTrajectory

from .robot import YAMRobot
from .scene import YAM_FINGERTIP_GRASP_SITE
from .wrapper import RobotWrapper, SimWrapper

from envs.common.robot_utils import rotation_6d_to_matrix, T_gripper
# from i2rt.robots.kinematics import Kinematics

# Default camera position for fixed camera
DEFAULT_CAMERA_POSITION = np.array([-0.7, 0.0, 0.6])  # 0.7m in front along robot base X axis
DEFAULT_CAMERA_LOOKAT = np.array([0.0, 0.0, 0.5])  # Center of workspace


def create_camera_config(
    camera_type: Literal["fixed", "tracking"],
) -> FixedCamera | TrackingCamera:
    """Create camera configuration based on camera type.

    Args:
        camera_type: Type of camera - "fixed" or "tracking"

    Returns:
        Camera configuration instance
    """
    if camera_type == "fixed":
        return FixedCamera(
            position=DEFAULT_CAMERA_POSITION,
            lookat=DEFAULT_CAMERA_LOOKAT,
        )
    elif camera_type == "tracking":
        return TrackingCamera(
            target_site=YAM_FINGERTIP_GRASP_SITE,
            distance=0.5,
            azimuth=135.0,
            elevation=-30.0,
        )
    else:
        raise ValueError(f"Invalid camera_type: {camera_type}. Must be 'fixed' or 'tracking'")


class YAMTableTopEnv:
    """YAM arm tabletop manipulation environment (base environment).

    Provides a tabletop workspace with the YAM arm mounted on a table.
    Uses composition to wrap MJManagerBasedEnv.
    """

    def __init__(
        self,
        table_size: tuple[float, float, float] = (0.6, 0.4, 0.02),
        table_pos: tuple[float, float, float] = (0.0, 0.0, 0.4),
        table_friction: tuple[float, float, float] = (2.0, 0.1, 0.001),
        timestep: float = 0.002,
        integrator: mujoco.mjtIntegrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST,  # type: ignore[assignment]
        time_limit: float | None = None,
        max_steps: int | None = None,
        camera_type: Literal["fixed", "tracking"] = "tracking",
        observation_manager: ObservationManager[MJState, Any] | None = None,
        action_manager: ActionManager[np.ndarray, np.ndarray] | None = None,
        reward_manager: RewardManager[MJState] | None = None,
        termination_manager: TerminationManager[MJState] | None = None,
        fps: float | None = None,
        enable_rendering: bool = False,
        render_width: int = 1280,
        render_height: int = 720,
        action_mode: Literal["joints", "pose"] = "joints",
    ):
        """Initialize the YAM tabletop environment.

        Args:
            table_size: (L, W, H) dimensions of the table top.
            table_pos: (x, y, z) position of the table center.
            table_friction: (sliding, torsional, rolling) friction parameters.
            timestep: Simulation timestep in seconds.
            integrator: MuJoCo integrator type.
            time_limit: Maximum episode time in seconds (mutually exclusive with max_steps).
            max_steps: Maximum episode steps (mutually exclusive with time_limit).
            camera_type: Type of camera - "fixed" or "tracking". Used for camera observations and visualization.
            observation_manager: Custom observation manager. If None, uses identity.
            action_manager: Custom action manager. If None, uses identity.
            reward_manager: Custom reward manager. If None, uses zero reward.
            termination_manager: Custom termination manager. If None, uses timeout only.
            fps: Target frequency for actions in Hz. If None, actions are executed at simulation timestep frequency.
                When set, actions are provided at this frequency and interpolated over multiple simulation steps.
            enable_rendering: If True, enables offscreen rendering. Call render() to get frames.
            render_width: Width of rendered frames in pixels (only used if enable_rendering=True).
            render_height: Height of rendered frames in pixels (only used if enable_rendering=True).
            action_mode: Action mode - "joints" for joint angle actions (7 dims) or "pose" for end effector pose actions (10 dims: 3D pos + 6D rot + gripper).
        """
        
        assert action_mode == "joints", "Only joint mode is supported for now"
        
        # Create camera configuration
        self.camera = create_camera_config(camera_type)

        # Build the scene spec (allows child classes to customize before compilation)
        from .scene import build_table_scene_spec
        spec = build_table_scene_spec(
            table_size=table_size,
            table_pos=table_pos,
            table_friction=table_friction,
            timestep=timestep,
            integrator=integrator,
        )

        # Allow child classes to customize the spec (e.g., add objects)
        spec = self._customize_scene_spec(spec, table_size=table_size, table_pos=table_pos)

        # Compile the spec
        xml_str = spec.to_xml()
        model = spec.compile()

        self.model = model
        self._model_xml = xml_str
        self.robot = YAMRobot()
        self.timestep = timestep

        self._enable_rendering = enable_rendering
        self._render_width = render_width
        self._render_height = render_height
        self._renderer: mujoco.Renderer | None = None
        self._render_camera: mujoco.MjvCamera | None = None
        if enable_rendering:
            self._renderer = mujoco.Renderer(model, height=render_height, width=render_width)
            self._render_camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, self._render_camera)

        if fps is not None:
            action_interval = 1.0 / fps
            self._num_steps_per_action = int(round(action_interval / timestep))
            if self._num_steps_per_action < 1:
                raise ValueError(
                    f"fps ({fps} Hz) is too high for timestep ({timestep} s). "
                    f"Action interval ({action_interval} s) must be >= timestep."
                )
            self._fps = fps
        else:
            self._num_steps_per_action = 1
            self._fps = None

        self._previous_action: np.ndarray | None = None

        # Allow child classes to customize scene manager (e.g., for randomization)
        scene_manager = self._create_scene_manager(model)

        if observation_manager is None:
            observation_manager = IdentityObservationManager(raw_observation_space(model))
        if action_manager is None:
            action_manager = IdentityActionManager(raw_action_space(model))
        if reward_manager is None:
            reward_manager = ZeroRewardManager()
        if termination_manager is None:
            if time_limit is None and max_steps is None:
                max_steps = 1000
            # TimeoutTerminationManager now handles conversion from env steps to sim steps
            # When fps is provided, max_steps is interpreted as environment steps
            termination_manager = TimeoutTerminationManager(
                model=model,
                time_limit=time_limit,
                max_steps=max_steps,
                fps=fps,
            )

        self._env = MJManagerBasedEnv(
            scene_manager=scene_manager,
            observation_manager=observation_manager,
            action_manager=action_manager,
            reward_manager=reward_manager,
            termination_manager=termination_manager,
        )

        self.sim = SimWrapper(model, lambda: self.scene_manager.data, xml_str=self._model_xml)

        self.robot_base_body_name = "yam_link_1"
        self.gripper_body_name = "gripper0_eef"
        self._robots = None
        self._action_mode = action_mode

        # Initialize kinematics for IK when using pose mode
        # if self._action_mode == "pose":
        #     self.kinematics = Kinematics(model, site_name=YAM_FINGERTIP_GRASP_SITE)
        # else:
        self.kinematics = None

        # Robot base frame correction: YAM robot base frame has X and Y axes flipped
        # (180-degree rotation about Z axis) compared to the expected convention
        self._robot_base_frame_correction = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _customize_scene_spec(
        self,
        spec: mujoco.MjSpec,
        table_size: tuple[float, float, float],
        table_pos: tuple[float, float, float],
    ) -> mujoco.MjSpec:
        """Override this method in child classes to customize the scene spec before compilation.

        Args:
            spec: The base scene spec (table and robot)
            table_size: Table dimensions
            table_pos: Table position

        Returns:
            The customized spec (can be the same spec object, modified in place)
        """
        return spec

    def _create_scene_manager(self, model: mujoco.MjModel) -> MJSceneManager:
        """Override this method in child classes to customize scene manager creation.

        For example, child classes can return a RandomizedMJSceneManager for randomization.

        Args:
            model: The compiled MuJoCo model

        Returns:
            Scene manager instance (can be MJSceneManager or RandomizedMJSceneManager)
        """
        return MJSceneManager(model=model)

    def _reset_to_upright_pose(self) -> None:
        """Reset robot to upright pose with end effector at (0.2, 0, 0.5) and gripper fully open."""
        if self.kinematics is None:
            self.kinematics = Kinematics(self.model, site_name=YAM_FINGERTIP_GRASP_SITE)

        data = self.scene_manager.data
        current_q = data.qpos[:7].copy()

        # Compute target pose and solve IK
        ori = R.from_rotvec([0.0, np.pi, 0.0]).as_matrix()
        T_target = np.eye(4, dtype=np.float64)
        T_target[:3, :3] = ori
        T_target[:3, 3] = [0.2, 0, 0.5]

        _success, ik_solution = self.kinematics.ik(
            T_target, YAM_FINGERTIP_GRASP_SITE, init_q=data.qpos
        )

        # Create target configuration with gripper fully open
        target_q = np.zeros(7, dtype=np.float64)
        target_q[:6] = ik_solution[:6]
        target_q[6] = 0.041

        # Execute trajectory using forward kinematics
        trajectory = LinearJointTrajectory(
            start_q=current_q,
            target_q=target_q,
            joint_durations=np.array([0.5] * 6 + [1.0], dtype=np.float64),
        )

        num_steps = int(np.ceil(trajectory.duration / self.timestep))
        dt = trajectory.duration / num_steps

        for i in range(num_steps):
            traj_q = trajectory((i + 1) * dt)
            data.qpos[:7] = traj_q
            mujoco.mj_forward(self.model, data)

        # Set final position and settle with controls
        data.qpos[:7] = target_q
        mujoco.mj_forward(self.model, data)

        data.ctrl[:6] = target_q[:6]
        if self.model.nu > 6:
            data.ctrl[6] = 0.041

        for _ in range(50):
            data.ctrl[:6] = target_q[:6]
            if self.model.nu > 6:
                data.ctrl[6] = 0.041
            mujoco.mj_step(self.model, data)

    def reset(self, *, rng: np.random.Generator) -> tuple[Any, Any]:
        """Reset the environment to its initial state."""
        self._previous_action = None
        obs, extras = self._env.reset(rng=rng)

        self._reset_to_upright_pose()

        data = self.scene_manager.data
        state = np.empty((self.scene_manager.nstate,))
        mujoco.mj_getState(self.model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        obs, obs_extras = self._env.observation_manager.compute(state)
        if "observation" in extras:
            extras["observation"].update(obs_extras)

        return obs, extras

    def step(self, action: np.ndarray) -> tuple[Any, Any, Any, Any, Any]:
        """Execute one environment step with the given action.

        If action_mode is "pose", converts end effector pose to joint angles using IK.
        If fps is set, interpolates between previous and current actions over multiple
        simulation steps before returning the observation.
        """
        # Convert pose to joint angles if needed
        if self._action_mode == "pose":
            action = self._convert_pose_to_joints(action)

        if self._fps is None or self._num_steps_per_action == 1:
            return self._env.step(action)

        if self._previous_action is None:
            self._previous_action = action.copy()
            return self._env.step(action)

        interpolated_actions = linear_interpolate(
            start=self._previous_action,
            end=action,
            num_steps=self._num_steps_per_action,
        )

        final_obs = None
        final_reward = 0.0
        final_terminated = False
        final_truncated = False
        final_extras: dict[str, Any] = {}

        for interpolated_action in interpolated_actions:
            obs, reward, terminated, truncated, extras = self._env.step(interpolated_action)
            final_reward = reward
            final_obs = obs
            final_terminated = terminated
            final_truncated = truncated
            final_extras = extras
            if terminated or truncated:
                break

        self._previous_action = action.copy()
        return final_obs, final_reward, final_terminated, final_truncated, final_extras

    def _convert_pose_to_joints(self, action: np.ndarray) -> np.ndarray:
        """Convert end effector pose action to joint angles using IK.

        Args:
            action: Pose action as (3D pos, 6D rot, gripper) = 10 dims

        Returns:
            Joint action as (6 arm joints, 1 gripper joint) = 7 dims
        """
        if self.kinematics is None:
            raise ValueError("Kinematics not initialized. action_mode must be 'pose' to use pose actions.")

        pos, ori_6d, gripper = action[:3], action[3:9], action[-1]

        # Convert 6D rotation to rotation matrix (in robot base frame)
        ori = rotation_6d_to_matrix(ori_6d)

        # Build target pose in robot base frame
        T_target_base = np.eye(4)
        T_target_base[:3, :3] = ori
        T_target_base[:3, 3] = pos

        # Transform from robot base frame to world frame
        robot_base_pose = self.get_robot_base_pose()
        T_target_world = robot_base_pose @ T_target_base  # gripper in world frame
        T_target_world = T_target_world @ np.linalg.inv(T_gripper)  # gripper site in world frame

        # Get current full joint configuration for IK initialization
        # IK expects full qpos array (includes all DOFs: arm + gripper + block + etc.)
        current_qpos = self.scene_manager.data.qpos.copy()
        current_q = current_qpos[:7].copy()  # Extract arm joints for later use

        # Solve IK to convert end effector pose to joint angles
        success, ik_solution = self.kinematics.ik(
            T_target_world,
            YAM_FINGERTIP_GRASP_SITE,
            init_q=current_qpos,
        )

        if not success:
            # If IK fails, try with neutral configuration
            neutral_qpos = current_qpos.copy()
            neutral_qpos[:6] = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])
            success, ik_solution = self.kinematics.ik(
                T_target_world,
                YAM_FINGERTIP_GRASP_SITE,
                init_q=neutral_qpos,
            )

        if not success:
            # If IK still fails, use current joint angles (don't move)
            ik_solution = current_q[:6]

        # Convert gripper from [-1, 1] to [0.0, 0.041] range
        # -1 = open -> 0.041, 1 = closed -> 0.0
        gripper_value = np.clip(gripper, -1.0, 1.0)
        gripper_joint = 0.041 * (1.0 - gripper_value) / 2.0  # Map [-1, 1] to [0, 0.041]

        # Convert to joint angles (7 dims: 6 arm joints + 1 gripper joint)
        return np.concatenate([ik_solution[:6], [gripper_joint]])

    @property
    def scene_manager(self) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._env.scene_manager

    @property
    def data(self) -> mujoco.MjData:
        return self.scene_manager.data

    @property
    def body_names(self) -> list[str]:
        names = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                names.append(name)
        return names

    @property
    def n_actuators(self) -> int:
        return self.model.nu

    @property
    def n_dof(self) -> int:
        return self.robot.n_dof

    def get_body_xpos(self, body_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        return self.data.body(body_id).xpos.copy()

    def get_body_xmat(self, body_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        return self.data.body(body_id).xmat.reshape(3, 3).copy()

    def get_body_xquat(self, body_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        return self.data.body(body_id).xquat.copy()

    def get_robot_base_pose(self) -> np.ndarray:
        """Get the robot base pose as a 4x4 homogeneous transformation matrix.

        Returns:
            4x4 homogeneous transformation matrix representing the robot base pose
            in the world frame. The rotation matrix is in the top-left 3x3 block,
            and the position is in the top-right 3x1 column.
        """
        base_pos = self.get_body_xpos(self.robot_base_body_name)
        base_ori = self.get_body_xmat(self.robot_base_body_name)

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = base_ori
        pose[:3, 3] = base_pos
        return pose

    def get_robot_eef_pose(self) -> np.ndarray:
        state = np.empty((self.scene_manager.nstate,))
        mujoco.mj_getState(
            self.model,
            self.scene_manager.data,
            state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )

        data = mjData_from_mjState(self.model, state)

        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, YAM_FINGERTIP_GRASP_SITE
        )
        if site_id < 0:
            site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "yam_grasp_site"
            )

        if site_id < 0:
            raise ValueError("Could not find end-effector site in model")

        site_xpos = data.site(site_id).xpos
        site_xmat = data.site(site_id).xmat.reshape(3, 3)

        pose = np.eye(4)
        pose[:3, :3] = site_xmat
        pose[:3, 3] = site_xpos
        return pose

    def transform_to_robot_base_frame(self, pose_matrix: np.ndarray) -> np.ndarray:
        """Transform a pose matrix from world frame to corrected robot base frame.

        This method transforms a pose from world frame to robot base frame and applies
        the robot base frame correction (180-degree rotation about Z axis) to align
        with the expected coordinate convention.

        Args:
            pose_matrix: 4x4 homogeneous transformation matrix in world frame

        Returns:
            4x4 homogeneous transformation matrix in corrected robot base frame
        """
        robot_base_pose = self.get_robot_base_pose()
        pose_in_base = np.linalg.inv(robot_base_pose) @ pose_matrix
        # Apply robot base frame correction
        pose_corrected = self._robot_base_frame_correction @ pose_in_base
        return pose_corrected

    @property
    def robot_base_frame_correction(self) -> np.ndarray:
        """Get the robot base frame correction matrix.

        Returns:
            4x4 transformation matrix that corrects the robot base frame orientation
            (180-degree rotation about Z axis to flip X and Y axes)
        """
        return self._robot_base_frame_correction.copy()

    @property
    def robots(self) -> list[RobotWrapper]:
        if self._robots is None:
            # Verify robot base body name exists in model
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.robot_base_body_name)
            if body_id < 0:
                raise ValueError(f"Robot base body '{self.robot_base_body_name}' not found in model")
            self._robots = [
                RobotWrapper(
                    self.model,
                    lambda: self.scene_manager.data,
                    self.robot_base_body_name,
                )
            ]
        return self._robots

    def render(self) -> np.ndarray | None:
        """Render the current scene and return an RGB image.

        Returns:
            RGB image as (height, width, 3) uint8 array, or None if rendering is disabled.
        """
        if not self._enable_rendering or self._renderer is None or self._render_camera is None:
            return None

        self.camera.configure_camera(self.model, self.data, self._render_camera)
        self._renderer.update_scene(self.data, camera=self._render_camera)  # pyright: ignore[reportUnknownMemberType]
        frame = self._renderer.render()  # pyright: ignore[reportUnknownMemberType]
        return frame.copy()

    def close(self) -> None:
        """Clean up rendering resources."""
        if self._renderer is None:
            return

        import gc

        renderer = self._renderer
        self._renderer = None
        self._render_camera = None

        def noop_del(self):  # type: ignore[misc]
            pass

        gl_context_class = None
        if hasattr(renderer, "_gl_context"):
            gl_context = getattr(renderer, "_gl_context", None)
            if gl_context is not None:
                gl_context_class = gl_context.__class__

        try:
            renderer.close()
        except Exception:
            pass

        try:
            renderer.__class__.__del__ = noop_del  # type: ignore[assignment]
            if gl_context_class is not None:
                gl_context_class.__del__ = noop_del  # type: ignore[assignment]
        except Exception:
            pass

        del renderer
        gc.collect()


class ZeroRewardManager:
    """Reward manager that always returns zero reward."""

    def compute(self, scene_state: MJState) -> tuple[float, dict[str, Any]]:
        return 0.0, {}

