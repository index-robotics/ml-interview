"""ML Interview environments package.

This package provides MuJoCo-based environments for robot manipulation tasks,
including the YAM tabletop manipulation environment suite.
"""

# Core environment components
from .mj_env import (
    MJManagerBasedEnv,
    MJSceneManager,
    MJState,
    TimeoutTerminationManager,
    mjData_from_mjState,
    raw_action_space,
    raw_observation_space,
)
from .managers import (
    ActionManager,
    IdentityActionManager,
    IdentityObservationManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from .types import CameraObservation, Extras, ObsDict, Reward, Terminated, Truncated

# YAM tabletop environments
from .yam_table_top import (
    YAMTableTopEnv,
    YAMPickAndPlaceBlockEnv,
    DEFAULT_PLACE_POS,
    YAMRobot,
    YAM_FINGERTIP_GRASP_SITE,
    create_camera_config,
    create_table_scene,
    Block,
    Cylinder,
    Sphere,
    TabletopObject,
)

__all__ = [
    # Core
    "MJManagerBasedEnv",
    "MJSceneManager",
    "MJState",
    "TimeoutTerminationManager",
    "mjData_from_mjState",
    "raw_action_space",
    "raw_observation_space",
    # Managers
    "ActionManager",
    "IdentityActionManager",
    "IdentityObservationManager",
    "ObservationManager",
    "RewardManager",
    "TerminationManager",
    # Types
    "CameraObservation",
    "Extras",
    "ObsDict",
    "Reward",
    "Terminated",
    "Truncated",
    # YAM environments
    "YAMTableTopEnv",
    "YAMPickAndPlaceBlockEnv",
    "DEFAULT_PLACE_POS",
    "YAMRobot",
    "YAM_FINGERTIP_GRASP_SITE",
    "create_camera_config",
    "create_table_scene",
    # Objects
    "TabletopObject",
    "Block",
    "Sphere",
    "Cylinder",
]

