"""Object definitions for YAM tabletop manipulation tasks."""

from typing import Any

import mujoco


class TabletopObject:
    """Base class for tabletop manipulation objects."""

    def __init__(
        self,
        name: str,
        size: tuple[float, float, float],
        mass: float = 0.05,
        rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
        friction: tuple[float, float, float] = (2.0, 0.1, 0.001),
        geom_type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_BOX,  # type: ignore[assignment]
    ):
        """Initialize a tabletop object.

        Args:
            name: Unique name for the object.
            size: (L, W, H) dimensions of the object.
            mass: Mass of the object in kg.
            rgba: (R, G, B, A) color values.
            friction: (sliding, torsional, rolling) friction parameters.
            geom_type: MuJoCo geometry type.
        """
        self.name = name
        self.size = size
        self.mass = mass
        self.rgba = rgba
        self.friction = friction
        self.geom_type = geom_type

    def add_to_spec(self, spec: mujoco.MjSpec, parent_body: Any, pos: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Any:
        """Add this object to a MuJoCo spec.

        Args:
            spec: MuJoCo spec to add the object to.
            parent_body: Parent body to attach the object to.
            pos: (x, y, z) position relative to parent body.

        Returns:
            The created body containing the object.
        """
        # Create body for the object
        obj_body = parent_body.add_body()
        obj_body.name = self.name
        obj_body.pos = list(pos)

        # Add geometry
        obj_geom = obj_body.add_geom()
        obj_geom.name = f"{self.name}_geom"
        obj_geom.type = self.geom_type
        obj_geom.size = [s / 2.0 for s in self.size]  # MuJoCo uses half-sizes
        obj_geom.rgba = list(self.rgba)
        obj_geom.mass = self.mass
        obj_geom.friction = list(self.friction)

        # Add free joint to allow the object to move
        obj_joint = obj_body.add_freejoint()
        obj_joint.name = f"{self.name}_freejoint"

        # Add a site marker for visualization
        obj_site = obj_body.add_site()
        obj_site.name = f"{self.name}_site"
        obj_site.type = mujoco.mjtGeom.mjGEOM_SPHERE
        obj_site.size = [0.01, 0.01, 0.01]
        obj_site.rgba = [1, 1, 0, 1]  # Yellow marker
        obj_site.pos = [0, 0, 0]

        return obj_body


class Block(TabletopObject):
    """A simple block object for manipulation tasks."""

    def __init__(
        self,
        name: str = "block",
        size: tuple[float, float, float] = (0.025, 0.025, 0.025),
        mass: float = 0.05,
        rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
    ):
        """Initialize a block object.

        Args:
            name: Name of the block.
            size: (L, W, H) dimensions in meters.
            mass: Mass in kg.
            rgba: (R, G, B, A) color values.
        """
        super().__init__(
            name=name,
            size=size,
            mass=mass,
            rgba=rgba,
            friction=(2.0, 0.1, 0.001),  # High friction for better grasping
            geom_type=mujoco.mjtGeom.mjGEOM_BOX,
        )


class Sphere(TabletopObject):
    """A simple sphere object for manipulation tasks."""

    def __init__(
        self,
        name: str = "sphere",
        radius: float = 0.02,
        mass: float = 0.05,
        rgba: tuple[float, float, float, float] = (0.2, 0.8, 0.2, 1.0),
    ):
        """Initialize a sphere object.

        Args:
            name: Name of the sphere.
            radius: Radius in meters.
            mass: Mass in kg.
            rgba: (R, G, B, A) color values.
        """
        super().__init__(
            name=name,
            size=(radius * 2, radius * 2, radius * 2),  # For sphere, size is diameter
            mass=mass,
            rgba=rgba,
            friction=(1.0, 0.1, 0.001),
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
        )


class Cylinder(TabletopObject):
    """A simple cylinder object for manipulation tasks."""

    def __init__(
        self,
        name: str = "cylinder",
        radius: float = 0.02,
        height: float = 0.05,
        mass: float = 0.05,
        rgba: tuple[float, float, float, float] = (0.2, 0.2, 0.8, 1.0),
    ):
        """Initialize a cylinder object.

        Args:
            name: Name of the cylinder.
            radius: Radius in meters.
            height: Height in meters.
            mass: Mass in kg.
            rgba: (R, G, B, A) color values.
        """
        super().__init__(
            name=name,
            size=(radius, radius, height / 2),  # MuJoCo cylinder: [radius, radius, half-height]
            mass=mass,
            rgba=rgba,
            friction=(1.5, 0.1, 0.001),
            geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        )

