"""Wrapper classes for MuJoCo model and data to provide robosuite-style API compatibility."""

# Mujoco typing preamble: incomplete types from mujoco.
# pyright: reportUnknownMemberType=none
# pyright: reportAttributeAccessIssue=none

import os
import tempfile
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np


class DataWrapper:
    """Wrapper around MuJoCo data to add convenience methods."""

    def __init__(self, model: mujoco.MjModel, data_getter: Callable[[], mujoco.MjData]):
        """Initialize data wrapper.

        Args:
            model: MuJoCo model
            data_getter: Callable that returns the current MuJoCo data
        """
        self._model = model
        self._data_getter = data_getter

    def _get_data(self) -> mujoco.MjData:
        """Get the current data and ensure forward kinematics are computed."""
        data = self._data_getter()
        # Ensure forward kinematics are computed (mj_step does this, but we ensure it here too)
        # This is safe to call multiple times and ensures data is up to date
        mujoco.mj_forward(self._model, data)
        return data

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying MuJoCo data."""
        # Special handling for geom_xpos and geom_xmat to return numpy arrays
        if name == "geom_xpos":
            data = self._get_data()
            return data.geom_xpos
        elif name == "geom_xmat":
            data = self._get_data()
            return data.geom_xmat
        return getattr(self._get_data(), name)

    def get_body_xpos(self, body_name: str) -> np.ndarray:
        """Get body position in world frame.

        Args:
            body_name: Name of the body.

        Returns:
            3D position array [x, y, z].
        """
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        data = self._get_data()
        return data.body(body_id).xpos.copy()

    def get_body_xmat(self, body_name: str) -> np.ndarray:
        """Get body rotation matrix in world frame.

        Args:
            body_name: Name of the body.

        Returns:
            3x3 rotation matrix.
        """
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        data = self._get_data()
        return data.body(body_id).xmat.reshape(3, 3).copy()


class ModelWrapper:
    """Wrapper around MuJoCo model to add convenience attributes."""

    def __init__(self, model: mujoco.MjModel, xml_str: str | None = None):
        """Initialize model wrapper.

        Args:
            model: MuJoCo model
            xml_str: Optional XML string representation of the model
        """
        self._model = model
        self._xml_str = xml_str

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying MuJoCo model."""
        # Special handling for robosuite-style attributes
        if name == "_geom_id2name":
            # Create mapping from geom ID to name
            geom_id2name = {}
            for i in range(self._model.ngeom):
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, i)
                geom_id2name[i] = name
            return geom_id2name
        elif name == "_body_id2name":
            # Create mapping from body ID to name
            body_id2name = {}
            for i in range(self._model.nbody):
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
                body_id2name[i] = name
            return body_id2name
        elif name == "_mesh_id2name":
            # Create mapping from mesh ID to name
            mesh_id2name = {}
            for i in range(self._model.nmesh):
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_MESH, i)
                mesh_id2name[i] = name
            return mesh_id2name
        return getattr(self._model, name)

    def get_xml(self) -> str:
        """Get the XML string representation of the model.

        Returns:
            XML string of the model.
        """
        # If we have the stored XML string, use it directly
        if self._xml_str is not None:
            return self._xml_str

        # Fallback: try to use mj_saveLastXML if model was loaded from XML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            temp_path = f.name
        try:
            mujoco.mj_saveLastXML(temp_path, self._model)
            with open(temp_path, 'r') as f:
                xml_str = f.read()
            return xml_str
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _extract_mj_names(
        self, name_adr: Any, num: int, obj_type: mujoco.mjtObj
    ) -> tuple[list[str], dict[str, int], dict[int, str]]:
        """Extract names from MuJoCo model name arrays.

        This is a compatibility method for robosuite-style code.

        Args:
            name_adr: Name address array from model
            num: Number of objects
            obj_type: Object type (mjOBJ_TEXTURE, mjOBJ_MATERIAL, etc.)

        Returns:
            Tuple of (names list, name_to_id dict, id_to_name dict)
        """
        names = []
        name_to_id = {}
        id_to_name = {}

        for i in range(num):
            name = mujoco.mj_id2name(self._model, obj_type, i)
            if name:
                names.append(name)
                name_to_id[name] = i
                id_to_name[i] = name

        return names, name_to_id, id_to_name

    @property
    def body_names(self) -> list[str]:
        """Get list of all body names in the model."""
        names = []
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:  # Filter out None/empty names
                names.append(name)
        return names

    def camera_name2id(self, camera_name: str) -> int:
        """Get camera ID from camera name (robosuite-style compatibility).

        Args:
            camera_name: Name of the camera.

        Returns:
            Camera ID.

        Raises:
            ValueError: If camera name not found.
        """
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")
        return cam_id

    def camera_id2name(self, camera_id: int) -> str | None:
        """Get camera name from camera ID (robosuite-style compatibility).

        Args:
            camera_id: ID of the camera.

        Returns:
            Camera name, or None if not found.
        """
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)


class SimWrapper:
    """Wrapper to provide robosuite-style sim.data and sim.model access."""

    def __init__(self, model: mujoco.MjModel, data_getter: Callable[[], mujoco.MjData], xml_str: str | None = None):
        """Initialize sim wrapper.

        Args:
            model: MuJoCo model
            data_getter: Callable that returns the current MuJoCo data
            xml_str: Optional XML string representation of the model
        """
        self._model_wrapper = ModelWrapper(model, xml_str=xml_str)
        self._data_wrapper = DataWrapper(model, data_getter)

    @property
    def model(self) -> ModelWrapper:
        """Access to MuJoCo model with convenience attributes."""
        return self._model_wrapper

    @property
    def data(self) -> DataWrapper:
        """Access to MuJoCo data with convenience methods."""
        return self._data_wrapper


class RobotWrapper:
    """Wrapper to provide robosuite-style robot interface with base_pos and base_ori."""

    def __init__(self, model: mujoco.MjModel, data_getter: Callable[[], mujoco.MjData], base_body_name: str):
        """Initialize robot wrapper.

        Args:
            model: MuJoCo model
            data_getter: Callable that returns the current MuJoCo data
            base_body_name: Name of the robot base body in the model
        """
        self._model = model
        self._data_getter = data_getter
        self._base_body_name = base_body_name
        self._base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        if self._base_body_id < 0:
            raise ValueError(f"Base body '{base_body_name}' not found in model")

    @property
    def base_pos(self) -> np.ndarray:
        """Get robot base position in world frame.

        Returns:
            3D position array [x, y, z].
        """
        data = self._data_getter()
        mujoco.mj_forward(self._model, data)
        return data.body(self._base_body_id).xpos.copy()

    @property
    def base_ori(self) -> np.ndarray:
        """Get robot base orientation in world frame.

        Returns:
            3x3 rotation matrix or 4D quaternion (depending on robosuite version).
        """
        data = self._data_getter()
        mujoco.mj_forward(self._model, data)
        # Return as rotation matrix (3x3) for compatibility
        # The code in robosuite_to_trimesh.py handles both formats
        return data.body(self._base_body_id).xmat.reshape(3, 3).copy()

