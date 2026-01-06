"""Camera observation manager for generating image observations."""

# pyright: reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false

from typing import final

import gymnasium as gym
import mujoco
import numpy as np

from envs.common.camera_config import CameraConfig, NamedCamera
from envs.mj_env import MJState, mjData_from_mjState
from envs.types import CameraObservation, Extras


@final
class CameraObservationManager:
    """Observation manager that generates camera images from MuJoCo simulation.

    This manager renders RGB images from a camera viewpoint at a specified FPS,
    caching frames between render intervals to reduce computational cost.

    The FPS must be chosen such that the frame period (1/fps) is divisible by
    the simulation timestep. This ensures frames are rendered at consistent
    simulation time intervals.

    Attributes:
        model: MuJoCo model for the simulation
        width: Image width in pixels
        height: Image height in pixels
        fps: Target frames per second for rendering
        camera_config: Camera configuration used for rendering
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        camera: str | CameraConfig,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ) -> None:
        """Initialize camera observation manager.

        Args:
            model: MuJoCo model
            camera: Camera name (string) or CameraConfig instance
            width: Image width in pixels
            height: Image height in pixels
            fps: Target frames per second for rendering

        Raises:
            ValueError: If FPS period is not divisible by model timestep
        """
        self.model = model
        self.width = width
        self.height = height
        self.fps = fps

        # Validate FPS is compatible with timestep
        timestep = model.opt.timestep
        self._frame_interval = 1.0 / fps

        # Check if frame_interval is a multiple of timestep (within floating point tolerance)
        ratio = self._frame_interval / timestep
        if not np.isclose(ratio, round(ratio), rtol=1e-6):
            raise ValueError(
                f"FPS period ({self._frame_interval:.6f}s) must be divisible by "
                + f"simulation timestep ({timestep:.6f}s). "
                + f"Ratio is {ratio:.6f}, should be near an integer."
            )

        # Convert string camera name to NamedCamera config
        if isinstance(camera, str):
            self.camera_config = NamedCamera(name=camera)
        else:
            self.camera_config = camera

        # Initialize renderer
        self._renderer = mujoco.Renderer(model, height=height, width=width)

        # Initialize camera
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self._camera)

        # Frame tracking
        # Initialize to -frame_interval so first call always renders
        self._last_frame_time = -self._frame_interval
        self._last_frame: CameraObservation | None = None

    @property
    def observation_space(self) -> gym.spaces.Box:
        """The Gymnasium observation space for camera images.

        Returns:
            A Box space with shape (height, width, 3) and dtype uint8,
            representing RGB images with values in [0, 255].
        """
        return gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

    def compute(self, scene_state: MJState) -> tuple[CameraObservation, Extras]:
        """Generate camera observation from scene state.

        This method renders a new frame only when sufficient simulation time has
        elapsed since the last render (based on target FPS). Between renders,
        it returns the cached last frame.

        Args:
            scene_state: Current MuJoCo full physics state

        Returns:
            Tuple of:
                - Camera image as (height, width, 3) RGB uint8 array
                - Extras dict with 'rendered_new_frame' bool and 'sim_time' float
        """
        # Reconstruct data from state
        data = mjData_from_mjState(self.model, scene_state)
        current_time = data.time

        # Check if it's time to render a new frame
        time_since_last = current_time - self._last_frame_time
        should_render = time_since_last >= self._frame_interval

        if should_render:
            # Update camera configuration (for tracking cameras)
            self.camera_config.configure_camera(self.model, data, self._camera)

            # Render frame
            self._renderer.update_scene(data, camera=self._camera)
            frame = self._renderer.render()

            # Cache frame and update time
            self._last_frame = frame.copy()
            self._last_frame_time = current_time

            extras: Extras = {"rendered_new_frame": True, "sim_time": current_time}
        else:
            # Return cached frame
            extras = {"rendered_new_frame": False, "sim_time": current_time}

        # Return the cached frame (guaranteed to exist after first call)
        assert self._last_frame is not None
        return self._last_frame, extras
