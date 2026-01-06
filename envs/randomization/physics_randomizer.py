"""Protocol for physics randomizers."""

from typing import Protocol

import mujoco
import numpy as np


class MJPhysicsRandomizer(Protocol):
    """Protocol for randomizing physics state in MuJoCo.

    Randomizers are stateless with respect to random number generation - they
    receive a numpy random generator at reset time to ensure determinism.
    """

    def reset(
        self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator
    ) -> None:
        """Apply randomization to the MuJoCo data.

        This method modifies the data in-place. The caller is responsible for
        calling mj_forward() after all randomizers have been applied.

        Args:
            model: MuJoCo model
            data: MuJoCo data to randomize (modified in-place)
            rng: Numpy random generator for deterministic randomization
        """
        ...
