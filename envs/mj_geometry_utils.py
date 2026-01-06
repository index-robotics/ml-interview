"""
Mujoco geometry utilities.

Functional API for working with Mujoco frames, agnostic to environment structure.
Takes model & data (standard mujoco primitives) as input.
"""

import colorsys
import logging
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypedDict

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T
import trimesh
from numpy.typing import NDArray

from envs.common.geometry_utils import to_4x4

# Correction to go from OpenGL to typical CV reference frame.
# Road to hell is paved with bugs from this.
T_GL_FROM_CV = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def get_T_world_from_A(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    A_name: str,
    entity_type: Literal["body", "camera", "camera-gl", "geom"],
) -> NDArray[np.float64]:
    """
    Get the transformation matrix from frame A to world coordinates.

    MuJoCo's xpos/xmat represent the entity's pose in world frame, which is T_world_from_entity.
    Left-multiplying a point in A's local frame by this matrix gives the point in world frame.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        A_name: Name of the entity
        entity_type: Type of entity - either "body" or "camera" or "camera-gl" (sans opencv convention).

    Returns:
        4x4 transformation matrix T_world_from_A.
        For cameras, applies OpenCV axis correction (OpenGL to OpenCV convention).
    """
    if entity_type == "body":
        # Check if body exists and get ID
        try:
            body = model.body(A_name)
        except KeyError:
            raise ValueError(
                f"Body '{A_name}' not found in model. Available bodies: {[model.body(i).name for i in range(model.nbody)]}"
            )

        body_id = body.id
        t_A = data.xpos[body_id]
        R_A = data.xmat[body_id].reshape(3, 3)
        T_world_from_A = to_4x4(R_A, t_A)
    elif entity_type == "camera":
        # Check if camera exists and get ID
        try:
            cam = model.camera(A_name)
        except KeyError:
            raise ValueError(
                f"Camera '{A_name}' not found in model. Available cameras: {[model.camera(i).name for i in range(model.ncam)]}"
            )

        cam_id_A = cam.id
        t_AGL = data.cam_xpos[cam_id_A]
        R_AGL = data.cam_xmat[cam_id_A].reshape(3, 3)
        # Apply camera axis correction for OpenCV compatibility
        T_world_from_A = to_4x4(R_AGL, t_AGL) @ T_GL_FROM_CV
    elif entity_type == "camera-gl":
        # Check if camera exists and get ID
        try:
            cam = model.camera(A_name)
        except KeyError:
            raise ValueError(
                f"Camera '{A_name}' not found in model. Available cameras: {[model.camera(i).name for i in range(model.ncam)]}"
            )

        cam_id_A = cam.id
        t_AGL = data.cam_xpos[cam_id_A]
        R_AGL = data.cam_xmat[cam_id_A].reshape(3, 3)
        T_world_from_A = to_4x4(R_AGL, t_AGL)
    elif entity_type == "geom":
        # Check if geom exists and get ID
        try:
            geom = model.geom(A_name)
        except KeyError:
            raise ValueError(
                f"Geom '{A_name}' not found in model. Available geoms: {[model.geom(i).name for i in range(model.ngeom)]}"
            )

        geom_id = geom.id
        t_A = data.geom_xpos[geom_id]
        R_A = data.geom_xmat[geom_id].reshape((3, 3))
        T_world_from_A = to_4x4(R_A, t_A)
    else:
        raise ValueError(f"entity_type must be 'body' or 'camera', got {entity_type}")

    return T_world_from_A


def get_camera_intrinsics(
    model: mujoco.MjModel,
    camera_name: str,
    height: int,
    width: int,
) -> NDArray[np.floating]:
    """
    Get camera intrinsics matrix.

    Args:
        model: MuJoCo model
        camera_name: Name of the camera
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        3x3 camera intrinsics matrix
    """
    # Check if camera exists and get fovy
    try:
        cam = model.camera(camera_name)
    except KeyError:
        raise ValueError(
            f"Camera '{camera_name}' not found in model. Available cameras: {[model.camera(i).name for i in range(model.ncam)]}"
        )

    fovy = model.cam_fovy[cam.id]
    fovy_radians = np.radians(fovy)

    # Calculate aspect ratio
    aspect_ratio = width / height

    # For rectangular images, we need to calculate focal lengths differently
    # The vertical FOV (fovy) is given, so we calculate fy first
    fy = 0.5 * height / np.tan(fovy_radians / 2)

    # For the horizontal focal length, we need to consider the aspect ratio
    # The horizontal FOV (fovx) is related to the vertical FOV by: tan(fovx/2) = aspect_ratio * tan(fovy/2)
    fovx_radians = 2 * np.arctan(aspect_ratio * np.tan(fovy_radians / 2))
    fx = 0.5 * width / np.tan(fovx_radians / 2)

    camera_intrinsics = np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
    return camera_intrinsics


def compute_T_A_from_B(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    A_name: str,
    B_name: str,
    entity_type: Literal["body", "camera"],
) -> NDArray[np.float64]:
    """
    Get the transformation matrix from frame B to frame A.

    This computes T_A_from_B = inv(T_world_from_B) @ T_world_from_A.
    Left-multiplying a point in B's local frame by this matrix gives the point in A's frame.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        A_name: Name of the target frame
        B_name: Name of the source frame
        entity_type: Type of entity - either "body" or "camera"

    Returns:
        4x4 transformation matrix T_A_from_B.
        For cameras, applies OpenCV axis correction (OpenGL to OpenCV convention).
    """
    T_world_from_A = get_T_world_from_A(model, data, A_name, entity_type)
    T_world_from_B = get_T_world_from_A(model, data, B_name, entity_type)

    # T_A_from_B transforms from B to A: inv(T_world_from_A) @ T_world_from_B
    T_A_from_B = np.linalg.inv(T_world_from_A) @ T_world_from_B

    return T_A_from_B


def mjdepth_to_meters(
    model: mujoco.MjModel, depth: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Convert depth image from MuJoCo to meters.

    Args:
        model: MuJoCo model
        depth: Depth image as (H, W) array

    Returns:
        Depth image in meters as (H, W) array, preserving dtype
    """
    extent: float = model.stat.extent
    near: float = model.vis.map.znear * extent
    far: float = model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image


type GeomName = str
type MeshName = str
type BodyName = str


class GeomMetadata(TypedDict):
    geom_name: GeomName
    geom_id: int
    body_name: BodyName
    body_id: int
    geom_group: int
    geom_type: int
    geom_type_name: str
    geom_size: list[float]
    geom_mesh_id: int | None
    geom_mesh_name: str | None
    geom_mesh_scale: list[float] | None
    geom_mesh_pos_offset: list[float] | None
    geom_mesh_quat_offset: list[float] | None
    geom_mesh_path: str | None
    geom_mesh_ref_pos: list[float] | None
    geom_mesh_ref_quat: list[float] | None
    geom_material: str | None
    geom_rgba: list[float] | None
    geom_name_is_fake: bool


def _extract_mj_names(
    model: mujoco.MjModel,
    obj_type: Literal["geom", "body", "mesh", "material", "texture"],
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Super simple, adapted from https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/binding_utils.py
    We'd import it if robosuite wasn't broken.
    See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1127
    """

    match obj_type:
        case "geom":
            mj_obj_type = mujoco.mjtObj.mjOBJ_GEOM
            num_obj = model.ngeom
        case "body":
            mj_obj_type = mujoco.mjtObj.mjOBJ_BODY
            num_obj = model.nbody
        case "mesh":
            mj_obj_type = mujoco.mjtObj.mjOBJ_MESH
            num_obj = model.nmesh
        case "material":
            mj_obj_type = mujoco.mjtObj.mjOBJ_MATERIAL
            num_obj = model.nmat
        case "texture":
            mj_obj_type = mujoco.mjtObj.mjOBJ_TEXTURE
            num_obj = model.ntex

    id2name: dict[int, str] = {}
    name2id: dict[str, int] = {}
    for i in range(num_obj):
        name = mujoco.mj_id2name(model, mj_obj_type, i)
        name2id[name] = i
        id2name[i] = name

    return name2id, id2name


GEOM_TYPE_TO_STR = {
    # mujoco.mjtGeom.mjGEOM_PLANE : "plane",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
}


class MeshRefInfo(TypedDict):
    refpos: NDArray[np.float64]
    refquat: NDArray[np.float64] | NDArray[np.int32]


def _read_byte_string_until_zero(byte_string: Any, start_index: int = 0) -> str:
    """
    Simple reader.
    Read byte string until 0 termination and convert result to string.
    """
    result = bytearray()
    for i in range(start_index, len(byte_string)):
        byte = byte_string[i]
        if byte == 0:
            break
        result.append(byte)
    return result.decode("utf-8")


def _extract_mesh_frames_from_xml(model: mujoco.MjModel) -> dict[MeshName, MeshRefInfo]:
    """There's probably a more modern way of doing this, but I want byte-identical.
    XML parsing might mess that up compared to directly accessing the model..."""
    with TemporaryDirectory() as td:
        filename = os.path.join(td, "model.xml")
        mujoco.mj_saveLastXML(filename.encode(), model)  # pyright: ignore[reportArgumentType]
        xml_str = open(filename).read()
    xml_root = ET.fromstring(xml_str)

    def _str2_arr(s):
        return np.array([float(x) for x in s.strip().split(" ")], dtype=np.float64)

    mesh_name_to_ref_info: dict[str, MeshRefInfo] = {}
    for mesh in xml_root.iter("mesh"):
        mesh_name = mesh.get("name")
        assert mesh_name is not None, (
            "Found mesh asset without a name attribute. All meshes must have names."
        )
        ref_pos = mesh.get("refpos")
        ref_quat = mesh.get("refquat")
        ref_info: MeshRefInfo = {
            "refpos": _str2_arr(ref_pos) if ref_pos is not None else np.zeros(3),
            "refquat": _str2_arr(ref_quat)
            if ref_quat is not None
            else np.array([1, 0, 0, 0], dtype=np.int32),
        }
        mesh_name_to_ref_info[mesh_name] = ref_info
    return mesh_name_to_ref_info


def extract_geom_metadata_from_mjmodel(
    model: mujoco.MjModel, exclude_prefixes: tuple[str, ...]
) -> tuple[
    list[GeomMetadata],
    dict[GeomName, list[int]],
]:
    # Get some mappings for names.
    _, geom_id_to_name = _extract_mj_names(model, "geom")
    _, body_id_to_name = _extract_mj_names(model, "body")
    _, mesh_id_to_name = _extract_mj_names(model, "mesh")
    _, mat_id_to_name = _extract_mj_names(model, "material")

    # Grab the per-mesh reference frames from the XML.
    mesh_name_to_ref_info = _extract_mesh_frames_from_xml(model)

    # Assemble all geoms.
    geom_model_infos: list[GeomMetadata] = []
    for geom_id in range(model.ngeom):
        geom_name = geom_id_to_name[geom_id]

        if model.geom_group[geom_id] == 1:
            logging.debug(f"skipping geom {geom_name} because it is visual-only.")
            continue

        if any(geom_name.startswith(geom_prefix) for geom_prefix in exclude_prefixes):
            logging.debug(
                f"skipping geom {geom_name} because it is excluded explicitly."
            )
            continue

        assert geom_name is not None
        geom_name_is_fake = False

        body_id = model.geom_bodyid[geom_id]
        body_name = body_id_to_name[body_id]

        # Mesh-related optional details.
        geom_type_name = GEOM_TYPE_TO_STR[model.geom_type[geom_id]]
        match geom_type_name:
            case "mesh":
                # get mesh id and name
                geom_mesh_id = int(model.geom_dataid[geom_id])

                assert geom_mesh_id != -1
                geom_mesh_name = mesh_id_to_name[geom_mesh_id]
                geom_mesh_scale = list(model.mesh_scale[geom_mesh_id])
                geom_mesh_pos_offset = list(model.mesh_pos[geom_mesh_id])
                geom_mesh_quat_offset = list(model.mesh_quat[geom_mesh_id])

                # read mesh path from asset bytestring
                geom_mesh_path_start_ind = int(model.mesh_pathadr[geom_mesh_id])
                geom_mesh_path = _read_byte_string_until_zero(
                    byte_string=model.paths, start_index=geom_mesh_path_start_ind
                )

                # Get reference frame info for this mesh
                ref_info = mesh_name_to_ref_info[geom_mesh_name]
                geom_mesh_ref_pos = ref_info["refpos"].tolist()
                geom_mesh_ref_quat = ref_info["refquat"].tolist()
            case _:
                geom_mesh_id = None
                geom_mesh_name = None
                geom_mesh_scale = None
                geom_mesh_pos_offset = None
                geom_mesh_quat_offset = None
                geom_mesh_path = None
                geom_mesh_ref_pos = None
                geom_mesh_ref_quat = None

        # There is either an rgba or a material.
        geom_mat_id = int(model.geom_matid[geom_id])
        match geom_mat_id:
            case -1:
                geom_material = None
                geom_rgba = list(model.geom_rgba[geom_id])
            case _:
                geom_material = mat_id_to_name[geom_mat_id]
                geom_rgba = None

        geom_model_infos.append(
            {
                "geom_name": geom_name,
                "geom_id": int(geom_id),
                "body_name": body_name,
                "body_id": int(body_id),
                "geom_group": int(model.geom_group[geom_id]),
                "geom_type": int(model.geom_type[geom_id]),
                "geom_type_name": geom_type_name,
                "geom_size": list(model.geom_size[geom_id]),
                "geom_material": geom_material,
                "geom_rgba": geom_rgba,
                "geom_name_is_fake": geom_name_is_fake,
                # Mesh keys.
                "geom_mesh_id": geom_mesh_id,
                "geom_mesh_name": geom_mesh_name,
                "geom_mesh_scale": geom_mesh_scale,
                "geom_mesh_pos_offset": geom_mesh_pos_offset,
                "geom_mesh_quat_offset": geom_mesh_quat_offset,
                "geom_mesh_path": geom_mesh_path,
                "geom_mesh_ref_pos": geom_mesh_ref_pos,
                "geom_mesh_ref_quat": geom_mesh_ref_quat,
            }
        )

    # Assemble a mapping from body to geom lists.
    body_to_geom_info_inds: dict[BodyName, list[int]] = defaultdict(list)
    for i, info in enumerate(geom_model_infos):
        body_to_geom_info_inds[info["body_name"]].append(i)

    return geom_model_infos, dict(body_to_geom_info_inds)


class GeomPoseInfo(TypedDict):
    geom_name: str
    geom_id: int
    geom_pose: list[float]
    geom_name_is_fake: bool


def extract_geom_pose_info(
    data: mujoco.MjData,
    geom_metadata: GeomMetadata,
    T_final_from_world: NDArray[np.float64],
) -> GeomPoseInfo:
    # get geom pose in world frame
    geom_id = geom_metadata["geom_id"]
    t_geom = np.array(data.geom_xpos[geom_id])
    R_geom = np.array(data.geom_xmat[geom_id].reshape((3, 3)))
    T_world_from_geom = to_4x4(R_geom, t_geom)

    # Apply various mesh offsets.
    if geom_metadata["geom_type_name"] == "mesh":
        # correct the pose of the mesh
        # see: https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh

        # First apply the mesh offset transformation if it exists
        if geom_metadata["geom_mesh_pos_offset"]:
            t_offset_from_geom = np.array(geom_metadata["geom_mesh_pos_offset"])
            R_offset_from_geom = T.quat2mat(
                T.convert_quat(
                    np.array(geom_metadata["geom_mesh_quat_offset"]), to="xyzw"
                )
            )
            T_offset_from_geom = to_4x4(R_offset_from_geom, t_offset_from_geom)
            # use their inv for binary equality.
            T_world_from_geom = T_world_from_geom @ T.pose_inv(T_offset_from_geom)

        # Then apply the reference frame transformation
        t_ref_from_geom = np.array(geom_metadata["geom_mesh_ref_pos"])
        R_ref_from_geom = T.quat2mat(
            T.convert_quat(np.array(geom_metadata["geom_mesh_ref_quat"]), to="xyzw")
        )
        T_ref_from_geom = to_4x4(R_ref_from_geom, t_ref_from_geom)
        # use their inv for binary equality.
        T_world_from_geom = T_world_from_geom @ T.pose_inv(T_ref_from_geom)

    # Apply final transform.
    T_final_from_geom = T_final_from_world @ T_world_from_geom
    t_final_from_geom, R_final_from_geom = (
        T_final_from_geom[..., :3, 3],
        T_final_from_geom[..., :3, :3],
    )
    quat_final_from_geom = T.convert_quat(T.mat2quat(R_final_from_geom), to="wxyz")
    qpos_final_from_geom = np.concatenate([t_final_from_geom, quat_final_from_geom])

    return {
        "geom_name": geom_metadata["geom_name"],
        "geom_id": geom_metadata["geom_id"],
        "geom_pose": qpos_final_from_geom.tolist(),
        "geom_name_is_fake": geom_metadata["geom_name_is_fake"],
    }


def mujoco_geoms_to_trimesh_scene(
    geom_metadatas: list[GeomMetadata], geom_poses: list[GeomPoseInfo]
) -> trimesh.Scene:
    """Simple mapping to create a trimesh scene!"""
    assert len(geom_metadatas) == len(geom_poses)
    num_geoms = len(geom_metadatas)

    # create an entity per geom
    mesh_obstacles = []
    cuboid_obstacles = []
    capsule_obstacles = []
    cylinder_obstacles = []
    sphere_obstacles = []

    # housekeeping for convenience
    obstacles_by_name = dict()

    colors = [
        colorsys.hsv_to_rgb(h, s=1.0, v=0.8)
        for h in np.linspace(0, 1, num_geoms, endpoint=False)
    ]
    for ind in range(num_geoms):
        geom_metadata = geom_metadatas[ind]
        geom_pose_info = geom_poses[ind]
        geom_color = np.append(colors[ind], 1.0)

        geom_name = geom_metadata["geom_name"]
        assert geom_pose_info["geom_name"] == geom_name
        geom_type_name = geom_metadata["geom_type_name"]
        geom_size = geom_metadata["geom_size"]
        qpos_origin_from_geom = geom_pose_info["geom_pose"]

        T_origin_from_geom = T.make_pose(
            qpos_origin_from_geom[:3],
            T.quat2mat(T.convert_quat(np.array(qpos_origin_from_geom[3:]), to="xyzw")),
        )

        match geom_type_name:
            case "box":
                obstacle = trimesh.creation.box(
                    extents=[2.0 * geom_size[0], 2.0 * geom_size[1], 2.0 * geom_size[2]]
                )
                obstacle.apply_transform(T_origin_from_geom)
                obstacle.visual.face_colors = geom_color[:3]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                cuboid_obstacles.append(obstacle)
            case "mesh":
                mesh_file = geom_metadata["geom_mesh_path"]
                mesh_scale = geom_metadata["geom_mesh_scale"]
                obstacle = trimesh.load_mesh(mesh_file)
                obstacle.apply_scale(mesh_scale)
                obstacle.apply_transform(T_origin_from_geom)
                # obstacle.visual.face_colors = geom_color[:3]
                mesh_obstacles.append(obstacle)
            case "capsule":
                obstacle = trimesh.creation.capsule(
                    radius=geom_size[0], height=2.0 * geom_size[1]
                )
                obstacle.apply_transform(T_origin_from_geom)
                obstacle.visual.face_colors = geom_color[:3]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                capsule_obstacles.append(obstacle)
            case "cylinder":
                obstacle = trimesh.creation.cylinder(
                    radius=geom_size[0], height=(2.0 * geom_size[1])
                )
                obstacle.apply_transform(T_origin_from_geom)
                obstacle.visual.face_colors = geom_color[:3]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                cylinder_obstacles.append(obstacle)
            case "sphere":
                obstacle = trimesh.creation.icosphere(radius=geom_size[0])
                obstacle.apply_transform(T_origin_from_geom)
                obstacle.visual.face_colors = geom_color[:3]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                sphere_obstacles.append(obstacle)
            case _:
                raise Exception("got invalid geom type: {}".format(geom_type_name))

        # store the obstacle object by geom name
        obstacles_by_name[geom_name] = obstacle

    # construct trimesh scene from the obstacles
    all_meshes = (
        mesh_obstacles
        + cuboid_obstacles
        + capsule_obstacles
        + cylinder_obstacles
        + sphere_obstacles
    )
    scene = trimesh.Scene(all_meshes)

    return scene


def extract_object_mesh_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_name: str,
    in_base_frame: bool,
    robot_base_body_name: str,
) -> trimesh.Trimesh:
    """Extract the mesh of an object from a Mujoco model.

    Probably this should be decomposed so we don't have to extract each mesh each time.

    Args:
        model: The Mujoco model.
        data: The Mujoco data.
        object_name: The name of the object.
        in_base_frame: Whether to return the object in the base frame.
        robot_base_body_name: The name of the robot base body.

    Returns:
        The mesh of the object.
    """
    mj_geom_model_infos, mj_body_to_geom_info_inds = extract_geom_metadata_from_mjmodel(
        model=model, exclude_prefixes=("robot", "gripper", "mobilebase")
    )

    if in_base_frame:
        T_world_from_robotbase = get_T_world_from_A(
            model, data, A_name=robot_base_body_name, entity_type="body"
        )
        T_robotbase_from_world = np.linalg.inv(T_world_from_robotbase)
    else:
        T_robotbase_from_world = np.eye(4)
    object_geom_indices = mj_body_to_geom_info_inds[object_name]
    object_geoms = [mj_geom_model_infos[i] for i in object_geom_indices]
    object_poses = [
        extract_geom_pose_info(data, mj_geom_model_infos[i], T_robotbase_from_world)  # pyright: ignore[reportArgumentType]
        for i in object_geom_indices
    ]

    object_scene = mujoco_geoms_to_trimesh_scene(object_geoms, object_poses)
    meshes = list(object_scene.geometry.values())
    if len(meshes) == 0:
        raise ValueError(f"No valid meshes found for object {object_name}")
    elif len(meshes) == 1:
        object_mesh = meshes[0]
    else:
        object_mesh = trimesh.util.concatenate(meshes)

    return object_mesh
