"""Camera configuration classes for MuJoCo environments."""

from typing import Protocol

import mujoco
import numpy as np
from numpy.typing import NDArray


class CameraConfig(Protocol):
    """Protocol for camera configurations."""

    def configure_camera(
        self, model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera
    ) -> None:
        """Configure the MuJoCo camera view based on this config.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            camera: MuJoCo viewport camera to configure
        """
        ...


class NamedCamera:
    """Camera configuration using a named camera from the MuJoCo model."""

    def __init__(self, name: str) -> None:
        """Initialize named camera config.

        Args:
            name: Name of the camera in the MuJoCo model
        """
        self.name = name

    def configure_camera(
        self, model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera
    ) -> None:
        """Configure camera to use the named camera from the model.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            camera: MuJoCo viewport camera to configure
        """
        try:
            cam_id = model.camera(self.name).id
            mujoco.mjv_setFreeCamera(model, camera, cam_id)
        except KeyError:
            # Fallback to default free camera if named camera not found
            mujoco.mjv_defaultFreeCamera(model, camera)


class FixedCamera:
    """Fixed camera configuration with position and lookat point."""

    def __init__(
        self,
        position: NDArray[np.float64] | tuple[float, float, float],
        lookat: NDArray[np.float64] | tuple[float, float, float],
    ) -> None:
        """Initialize fixed camera config.

        Args:
            position: Camera position in world coordinates (x, y, z)
            lookat: Point to look at in world coordinates (x, y, z)
        """
        self.position = np.array(position, dtype=np.float64)
        self.lookat = np.array(lookat, dtype=np.float64)

    def configure_camera(
        self, model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera
    ) -> None:
        """Configure camera to fixed position and lookat.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            camera: MuJoCo viewport camera to configure
        """
        camera.lookat[:] = self.lookat
        camera.distance = np.linalg.norm(self.position - self.lookat)
        # Compute azimuth and elevation from position
        diff = self.position - self.lookat
        horizontal_dist = np.linalg.norm(diff[:2])
        camera.azimuth = np.degrees(np.arctan2(diff[1], diff[0]))
        camera.elevation = np.degrees(np.arctan2(diff[2], horizontal_dist))


class TrackingCamera:
    """Tracking camera that follows a target site."""

    def __init__(
        self,
        target_site: str,
        distance: float = 0.5,
        azimuth: float = 135.0,
        elevation: float = -30.0,
    ) -> None:
        """Initialize tracking camera config.

        Args:
            target_site: Name of the site to track
            distance: Distance from target in meters
            azimuth: Azimuth angle in degrees
            elevation: Elevation angle in degrees
        """
        self.target_site = target_site
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation

    def configure_camera(
        self, model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera
    ) -> None:
        """Configure camera to track the target site.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            camera: MuJoCo viewport camera to configure
        """
        try:
            site_id = model.site(self.target_site).id
            camera.lookat[:] = data.site_xpos[site_id]
            camera.distance = self.distance
            camera.azimuth = self.azimuth
            camera.elevation = self.elevation
        except KeyError:
            # Fallback to default if site not found
            mujoco.mjv_defaultFreeCamera(model, camera)

