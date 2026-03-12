"""Core scene creation for YAM arm tabletop manipulation environment (table and robot only)."""

# Mujoco typing preamble: incomplete types from mujoco.
# pyright: reportUnknownMemberType=none
# pyright: reportAttributeAccessIssue=none

import mujoco
import numpy as np
import robot_descriptions.yam_mj_description as yam_desc
from scipy.spatial.transform import Rotation

PLATE_OFFSET = 0.01
LEG_SIZE = [0.03, 0.2, 0.0]

YAM_FINGERTIP_GRASP_SITE = "yam_fingertip_grasp_site"


def build_table_scene_spec(
    table_size: tuple[float, float, float],
    table_pos: tuple[float, float, float],
    table_friction: tuple[float, float, float],
    timestep: float,
    integrator: mujoco.mjtIntegrator,
) -> mujoco.MjSpec:
    """Build the table scene spec (without compiling).

    This is a helper function that creates the spec for the base scene.
    It can be used by create_table_scene or by task environments that need
    to modify the spec before compilation.

    Args:
        table_size: (L, W, H) dimensions of the table top.
        table_pos: (x, y, z) position of the table center.
        table_friction: (sliding, torsional, rolling) friction parameters.
        timestep: Simulation timestep in seconds.
        integrator: MuJoCo integrator type.

    Returns:
        MuJoCo spec for the table scene.
    """
    spec = mujoco.MjSpec()
    spec.modelname = "yam_arm_table_scene"
    spec.compiler.autolimits = True
    spec.option.timestep = timestep
    spec.option.integrator = integrator

    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
    spec.visual.headlight.specular = [0.3, 0.3, 0.3]
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1]
    spec.visual.global_.azimuth = 120
    spec.visual.global_.elevation = -20
    spec.visual.global_.offwidth = 2560
    spec.visual.global_.offheight = 1440
    spec.visual.scale.framelength = 0.1
    spec.visual.scale.framewidth = 0.005

    skybox = spec.add_texture()
    skybox.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    skybox.rgb1 = [0.3, 0.5, 0.7]
    skybox.rgb2 = [0.0, 0.0, 0.0]
    skybox.width = 800
    skybox.height = 800
    skybox.mark = mujoco.mjtMark.mjMARK_RANDOM
    skybox.markrgb = [1, 1, 1]

    ground_tex = spec.add_texture()
    ground_tex.name = "groundplane"
    ground_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    ground_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    ground_tex.rgb1 = [0.2, 0.3, 0.4]
    ground_tex.rgb2 = [0.1, 0.2, 0.3]
    ground_tex.width = 300
    ground_tex.height = 300
    ground_tex.mark = mujoco.mjtMark.mjMARK_EDGE
    ground_tex.markrgb = [0.8, 0.8, 0.8]

    ground_mat = spec.add_material()
    ground_mat.name = "groundplane"
    ground_mat.textures = ["groundplane"] + [""] * 9
    ground_mat.texuniform = True
    ground_mat.texrepeat = [5, 5]
    ground_mat.reflectance = 0.0

    light1 = spec.worldbody.add_light()
    light1.pos = [0, 0, 1.5]
    light1.dir = [0, 0, -1]
    light1.castshadow = False

    light2 = spec.worldbody.add_light()
    light2.pos = [1, 1, 2]
    light2.dir = [-1, -1, -1]

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.material = "groundplane"
    floor.group = 1

    camera = spec.worldbody.add_camera()
    camera.name = "main_camera"
    camera.pos = [-0.7, 0.0, 0.6]
    target = np.array([0.0, 0.0, 0.5])
    direction = (target - camera.pos) / np.linalg.norm(target - camera.pos)
    z_axis = -direction
    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up, z_axis) / np.linalg.norm(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
    camera.quat = Rotation.from_matrix(rot_matrix).as_quat(scalar_first=True)

    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = list(table_pos)

    table_top = table.add_geom()
    table_top.name = "table_top"
    table_top.type = mujoco.mjtGeom.mjGEOM_BOX
    table_top.size = [s / 2.0 for s in table_size]
    table_top.rgba = [0.6, 0.4, 0.2, 1]
    table_top.pos = [0, 0, 0]
    table_top.friction = list(table_friction)

    leg_radius = LEG_SIZE[0]
    leg_height = LEG_SIZE[1]
    leg_size = [leg_radius, leg_radius, leg_height / 2.0]
    leg_positions = [
        (table_size[0] / 2 - 0.1, table_size[1] / 2 - 0.1),
        (table_size[0] / 2 - 0.1, -table_size[1] / 2 + 0.1),
        (-table_size[0] / 2 + 0.1, table_size[1] / 2 - 0.1),
        (-table_size[0] / 2 + 0.1, -table_size[1] / 2 + 0.1),
    ]

    for i, (x, y) in enumerate(leg_positions, 1):
        leg = table.add_geom()
        leg.name = f"table_leg{i}"
        leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        leg.size = leg_size
        leg.rgba = [0.5, 0.3, 0.15, 1]
        leg.pos = [x, y, -leg_height]

    arm_spec = mujoco.MjSpec.from_file(yam_desc.MJCF_PATH)
    link_6 = arm_spec.body("link_6")
    fingertip_site = link_6.add_site()
    fingertip_site.name = "fingertip_grasp_site"
    fingertip_site.pos = [0, -0.05, 0.1347]
    fingertip_site.quat = [1, 0, 0, -1]
    fingertip_site.size = [0.005, 0.005, 0.005]
    fingertip_site.rgba = [0, 0, 1, 1]
    fingertip_site.group = 4

    arm_mount = spec.worldbody.add_site()
    arm_mount.name = "arm_mount"
    arm_mount.pos = [table_pos[0], table_pos[1], table_pos[2] + table_size[2] / 2 + PLATE_OFFSET]
    spec.attach(arm_spec, site="arm_mount", prefix="yam_")

    # Add gripper0_eef body at the same pose as the fingertip site
    # Add it after attachment so it has the correct name without prefix
    yam_link_6 = spec.body("yam_link_6")
    gripper0_eef = yam_link_6.add_body()
    gripper0_eef.name = "gripper0_eef"
    gripper0_eef.pos = [0, -0.05, 0.1347]  # Same position as fingertip site
    gripper0_eef.quat = [1, 0, 0, -1]  # Same orientation as fingertip site
    # Add a small invisible geom so the body exists (bodies need at least one geom or joint)
    gripper0_eef_geom = gripper0_eef.add_geom()
    gripper0_eef_geom.name = "gripper0_eef_geom"
    gripper0_eef_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    gripper0_eef_geom.size = [0.001, 0.001, 0.001]  # Very small, invisible
    gripper0_eef_geom.rgba = [0, 0, 0, 0]  # Fully transparent
    gripper0_eef_geom.group = 1  # Visual group (won't affect collisions)
    gripper0_eef_geom.contype = 0  # No collision
    gripper0_eef_geom.conaffinity = 0  # No collision

    table_top_site = table.add_site()
    table_top_site.name = "table_top_site"
    table_top_site.pos = [0, 0, table_size[2] / 2]
    table_top_site.size = [0.01, 0.01, 0.01]
    table_top_site.rgba = [0, 1, 0, 0.5]
    table_top_site.type = mujoco.mjtGeom.mjGEOM_SPHERE

    return spec


def create_table_scene(
    table_size: tuple[float, float, float] = (0.6, 0.4, 0.02),
    table_pos: tuple[float, float, float] = (0.0, 0.0, 0.4),
    table_friction: tuple[float, float, float] = (2.0, 0.1, 0.001),
    timestep: float = 0.002,
    integrator: mujoco.mjtIntegrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST,  # type: ignore[assignment]
) -> tuple[mujoco.MjModel, str]:
    """Create a core scene with the YAM arm on a table (no objects).

    This creates the base scene with just the table and robot. Objects should be
    added by task-specific environments.

    Args:
        table_size: (L, W, H) dimensions of the table top.
        table_pos: (x, y, z) position of the table center.
        table_friction: (sliding, torsional, rolling) friction parameters.
        timestep: Simulation timestep in seconds.
        integrator: MuJoCo integrator type.

    Returns:
        Tuple of (compiled MuJoCo model, XML string).
    """
    spec = build_table_scene_spec(
        table_size=table_size,
        table_pos=table_pos,
        table_friction=table_friction,
        timestep=timestep,
        integrator=integrator,
    )
    xml_str = spec.to_xml()
    model = spec.compile()
    return model, xml_str

