"""Randomized scene manager wrapper."""

from collections.abc import Sequence
from typing import final

import mujoco
import numpy as np

from envs.mj_env import MJCtrl, MJSceneManager, MJState
from envs.randomization.physics_randomizer import MJPhysicsRandomizer
from envs.types import Extras


@final
class RandomizedMJSceneManager:
    """Scene manager wrapper that applies physics randomization at reset.

    This wrapper applies a list of physics randomizers to the scene after
    resetting to the initial state. Randomizers are applied in order.

    Example:
        >>> model = create_scene_with_table()
        >>> randomizers = [
        ...     MjFreeJointRandomizer("block", x_range=(-0.2, 0.2), y_range=(-0.1, 0.1))
        ... ]
        >>> scene = RandomizedMJSceneManager(MJSceneManager(model), randomizers)
    """

    def __init__(
        self,
        scene_manager: MJSceneManager,
        randomizers: Sequence[MJPhysicsRandomizer],
    ):
        """Initialize randomized scene manager.

        Args:
            scene_manager: SceneManager
            randomizers: List of physics randomizers to apply at reset
        """
        self.scene_manager = scene_manager
        self.randomizers = randomizers

    def step(self, scene_action: MJCtrl) -> tuple[MJState, Extras]:
        """Step the simulation (same as MJSceneManager).

        Args:
            scene_action: Control action to apply

        Returns:
            Tuple of (state, extras)
        """
        return self.scene_manager.step(scene_action)

    def reset(self, *, rng: np.random.Generator) -> tuple[MJState, Extras]:
        """Reset the scene and apply randomization.

        NOTE: This is a bit hacky / inelegant. Might want a different way of handling
        the randomization, it's a bit annoying that we have to do it this way and
        recreate the

        Returns:
            Tuple of (randomized_state, extras)
        """
        # Reset to keyframe 0 (initial positions from XML)
        mujoco.mj_resetData(self.scene_manager.model, self.scene_manager.data)
        mujoco.mj_resetDataKeyframe(
            self.scene_manager.model, self.scene_manager.data, 0
        )

        # Apply randomizers in order (before mj_forward)
        for randomizer in self.randomizers:
            randomizer.reset(self.scene_manager.model, self.scene_manager.data, rng)

        # Update derived quantities after randomization
        mujoco.mj_forward(self.scene_manager.model, self.scene_manager.data)

        # Extract state
        state = np.empty((self.scene_manager.nstate,))
        mujoco.mj_getState(
            self.scene_manager.model,
            self.scene_manager.data,
            state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        return state, {}

    @property
    def model(self) -> mujoco.MjModel:
        """Access to the MuJoCo model."""
        return self.scene_manager.model

    @property
    def data(self) -> mujoco.MjData:
        """Access to the MuJoCo data."""
        return self.scene_manager.data

    @property
    def nstate(self) -> int:
        """Size of the full physics state vector."""
        return self.scene_manager.nstate

    @property
    def control_period(self) -> float:
        """Time between control updates in seconds.

        Delegates to the wrapped scene manager's control period.
        """
        return self.scene_manager.control_period
