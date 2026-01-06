"""Very clearly inspired by the IsaacLab managers (and mjlab),
but customized for my own purposes. We may want to switch to mjlab or others once
they stabilize, but for now we aren't really looking for 'vectorized'."""

from collections.abc import Mapping
from typing import Any, Generic, Protocol, TypeVar, final

import gymnasium as gym
import numpy as np

from envs.types import Extras, ObsDict, Reward, Terminated, Truncated

SceneState_co = TypeVar("SceneState_co", covariant=True)
SceneState_contra = TypeVar("SceneState_contra", contravariant=True)
SceneAction_co = TypeVar("SceneAction_co", covariant=True)
SceneAction_contra = TypeVar("SceneAction_contra", contravariant=True)
Observation_co = TypeVar("Observation_co", covariant=True)
Observation_contra = TypeVar("Observation_contra", contravariant=True)
Action_co = TypeVar("Action_co", covariant=True)
Action_contra = TypeVar("Action_contra", contravariant=True)
Action_inv = TypeVar("Action_inv")

##################################################################
# Basic protocols for the managers.
##################################################################


class SceneManager(Protocol[SceneState_co, SceneAction_contra]):
    """Protocol for managing the low-level physics simulation or scene state.

    The SceneManager is responsible for maintaining and advancing the underlying
    simulation state. It provides the fundamental reset and step operations that
    drive the physics engine or scene representation.

    Type Parameters:
        SceneState_co: The covariant scene state type (e.g., MJState for MuJoCo).
        SceneAction_contra: The contravariant action type consumed by the scene
            (e.g., MJCtrl for MuJoCo control inputs).
    """

    def reset(self, *, rng: np.random.Generator) -> tuple[SceneState_co, Extras]:
        """Reset the scene to its initial state.

        Returns:
            A tuple containing:
                - The initial scene state after reset
                - A dictionary of extra information (e.g., debug data)
        """
        ...

    def step(self, scene_action: SceneAction_contra) -> tuple[SceneState_co, Extras]:
        """Advance the scene by one simulation step given an action.

        Args:
            scene_action: The low-level action to apply to the scene.

        Returns:
            A tuple containing:
                - The new scene state after stepping
                - A dictionary of extra information (e.g., collision data)
        """
        ...

    @property
    def control_period(self) -> float:
        """The control period (in seconds) between scene control updates.

        This is an abstract property that implementations of SceneManager must
        provide.
        """
        ...


class ActionManager(Protocol[Action_inv, SceneAction_co]):
    """Protocol for transforming high-level actions into scene-level actions.

    The ActionManager acts as an adapter between the environment's action space
    (what the agent sees) and the scene's action space (what the physics engine
    expects). This allows for action preprocessing, normalization, or complex
    mappings between action representations.

    Type Parameters:
        Action_inv: The invariant high-level action type from the agent.
        SceneAction_co: The covariant low-level action type for the scene.

    Examples:
        - Denormalizing actions from [-1, 1] to joint limits
        - Mapping discrete actions to continuous controls
        - Applying action filters or smoothing
    """

    @property
    def action_space(self) -> gym.Space[Action_inv]:
        """The Gymnasium action space for this manager.

        Returns:
            A Gymnasium Space object describing the structure and bounds of
            actions consumed by this manager.
        """
        ...

    def compute(self, action: Action_inv) -> tuple[SceneAction_co, Extras]:
        """Transform a high-level action into a scene-level action.

        Args:
            action: The high-level action from the agent.

        Returns:
            A tuple containing:
                - The transformed scene-level action
                - A dictionary of extra information (e.g., clipping info)
        """
        ...


class ObservationManager(Protocol[SceneState_contra, Observation_co]):
    """Protocol for transforming scene state into agent observations.

    The ObservationManager processes raw scene state into the observation space
    that the agent interacts with. This enables feature extraction, state
    filtering, sensor simulation, or custom observation representations.

    Type Parameters:
        SceneState_contra: The contravariant scene state type to process.
        Observation_co: The covariant observation type for the agent.

    Examples:
        - Extracting joint positions and velocities from full physics state
        - Computing proprioceptive and exteroceptive features
        - Simulating noisy sensors or partial observability
        - Building observation histories or stacking frames
    """

    @property
    def observation_space(self) -> gym.Space[Observation_co]:
        """The Gymnasium observation space for this manager.

        Returns:
            A Gymnasium Space object describing the structure and bounds of
            observations produced by this manager.
        """
        ...

    def compute(self, scene_state: SceneState_contra) -> tuple[Observation_co, Extras]:
        """Transform scene state into an agent observation.

        Args:
            scene_state: The current scene state to process.

        Returns:
            A tuple containing:
                - The computed observation for the agent
                - A dictionary of extra information (e.g., raw sensor data)
        """
        ...


class RewardManager(Protocol[SceneState_contra]):
    """Protocol for computing rewards from scene state.

    The RewardManager evaluates the current scene state and produces a scalar
    reward signal for the agent. This can involve computing task progress,
    penalizing undesired behaviors, or combining multiple reward terms.

    Type Parameters:
        SceneState_contra: The contravariant scene state type to evaluate.

    Examples:
        - Distance-based rewards for reaching targets
        - Velocity tracking rewards
        - Penalty terms for constraint violations
        - Composite rewards with multiple weighted components
    """

    def compute(self, scene_state: SceneState_contra) -> tuple[Reward, Extras]:
        """Compute the reward for the current scene state.

        Args:
            scene_state: The current scene state to evaluate.

        Returns:
            A tuple containing:
                - The scalar reward value
                - A dictionary of extra information (e.g., reward components)
        """
        ...


class TerminationManager(Protocol[SceneState_contra]):
    """Protocol for determining episode termination and truncation.

    The TerminationManager decides whether an episode should end, distinguishing
    between natural termination (task success/failure) and truncation (timeout
    or external limits). This separation is important for proper bootstrapping
    in RL algorithms.

    Type Parameters:
        SceneState_contra: The contravariant scene state type to evaluate.

    Note:
        Following Gymnasium conventions:
        - Terminated: Episode ends due to task completion or failure
        - Truncated: Episode ends due to time limits or external constraints

    Examples:
        - Terminating when goal is reached or robot falls
        - Truncating after maximum episode steps
        - Composite termination from multiple conditions
    """

    def compute(
        self, scene_state: SceneState_contra, reward: Reward
    ) -> tuple[Terminated, Truncated, Extras]:
        """Determine if the episode should terminate or truncate.

        Args:
            scene_state: The current scene state to evaluate.
            reward: The current reward value (may influence termination).

        Returns:
            A tuple containing:
                - terminated: True if episode ends naturally (success/failure)
                - truncated: True if episode ends due to limits (timeout)
                - A dictionary of extra information (e.g., termination reason)
        """
        ...


##################################################################
# Helpful implementations.
##################################################################

ActionT = TypeVar("ActionT")
ObservationT = TypeVar("ObservationT")


@final
class IdentityActionManager(Generic[ActionT]):
    """Identity action manager that passes actions through unchanged.

    This implementation is useful when the agent's action space matches
    the scene's action space exactly, requiring no transformation. It
    serves as a default or placeholder when action preprocessing is not needed.

    Type Parameters:
        ActionT: The action type (same for both input and output).

    Attributes:
        action_space: The Gymnasium space for actions.
    """

    def __init__(self, action_space: gym.Space[ActionT]):
        """Initialize the identity action manager.

        Args:
            action_space: The Gymnasium space describing the actions.
        """
        self._action_space = action_space

    @property
    def action_space(self) -> gym.Space[ActionT]:
        """The Gymnasium action space for this manager."""
        return self._action_space

    def compute(self, action: ActionT) -> tuple[ActionT, Extras]:
        """Return the action unchanged.

        Args:
            action: The action to pass through.

        Returns:
            A tuple containing:
                - The same action unchanged
                - An empty extras dictionary
        """
        return action, {}


@final
class IdentityObservationManager(Generic[ObservationT]):
    """Identity observation manager that passes scene state through unchanged.

    This implementation is useful when the scene state is already in the desired
    observation format, requiring no feature extraction or transformation. It
    serves as a default when the raw scene state is the observation.

    Type Parameters:
        ObservationT: The scene state/observation type (same for both).

    Attributes:
        observation_space: The Gymnasium space for observations.
    """

    def __init__(self, observation_space: gym.Space[ObservationT]):
        """Initialize the identity observation manager.

        Args:
            observation_space: The Gymnasium space describing the observations.
        """
        self._observation_space = observation_space

    @property
    def observation_space(self) -> gym.Space[ObservationT]:
        """The Gymnasium observation space for this manager."""
        return self._observation_space

    def compute(self, scene_state: ObservationT) -> tuple[ObservationT, Extras]:
        """Return the scene state unchanged as the observation.

        Args:
            scene_state: The scene state to pass through.

        Returns:
            A tuple containing:
                - The same scene state unchanged
                - An empty extras dictionary
        """
        return scene_state, {}


@final
class RewardThresholdTerminationManager(Generic[SceneState_contra]):
    """Termination manager that ends episodes when reward reaches a threshold.

    This manager terminates episodes successfully when the reward meets or
    exceeds a specified threshold. Useful for tasks with clear success criteria
    expressed as reward values.

    Type Parameters:
        SceneState_contra: The contravariant scene state type (unused by this manager).

    Attributes:
        reward_threshold: The reward value at or above which to terminate.
    """

    def __init__(self, reward_threshold: float):
        """Initialize the reward threshold termination manager.

        Args:
            reward_threshold: Episodes terminate when reward >= this value.
        """
        self.reward_threshold = reward_threshold

    def compute(
        self,
        scene_state: SceneState_contra,  # pyright: ignore[reportUnusedParameter]
        reward: Reward,
    ) -> tuple[Terminated, Truncated, Extras]:
        """Check if the reward meets the termination threshold.

        Args:
            scene_state: The current scene state (unused).
            reward: The current reward value to check.

        Returns:
            A tuple containing:
                - terminated: True if reward >= threshold, False otherwise
                - truncated: Always False (this manager never truncates)
                - An empty extras dictionary
        """
        terminated = reward >= self.reward_threshold
        truncated = False
        return terminated, truncated, {"task_completed": terminated}


@final
class CompositeTerminationManager(Generic[SceneState_contra]):
    """Termination manager that combines multiple termination conditions with OR logic.

    This manager evaluates multiple child termination managers and terminates
    or truncates if ANY of them signal termination/truncation. This is useful
    for combining multiple ending conditions (e.g., timeout + success + failure).

    The extras dictionary preserves the individual results from each child manager,
    allowing inspection of which condition(s) triggered.

    Type Parameters:
        SceneState_contra: The contravariant scene state type.

    Attributes:
        termination_managers: Mapping of names to termination manager instances.
    """

    def __init__(
        self,
        termination_managers: Mapping[str, TerminationManager[SceneState_contra]],
    ):
        """Initialize the composite termination manager.

        Args:
            termination_managers: A mapping from descriptive names to termination
                manager instances. Each manager will be evaluated on every step.
        """
        self.termination_managers = termination_managers

    def compute(
        self, scene_state: SceneState_contra, reward: Reward
    ) -> tuple[Terminated, Truncated, Extras]:
        """Evaluate all child managers and combine results with OR logic.

        Args:
            scene_state: The current scene state to evaluate.
            reward: The current reward value.

        Returns:
            A tuple containing:
                - terminated: True if ANY child manager terminates
                - truncated: True if ANY child manager truncates
                - extras: Dict mapping manager names to their individual results
                    (terminated, truncated, extras) tuples
        """
        terminated = False
        truncated = False
        extras: dict[str, tuple[Terminated, Truncated, Extras]] = {}
        for manager_name, manager in self.termination_managers.items():
            terminated_, truncated_, extras_ = manager.compute(scene_state, reward)
            terminated = terminated or terminated_
            truncated = truncated or truncated_
            extras[manager_name] = terminated_, truncated_, extras_

        return terminated, truncated, extras


@final
class CompositeObservationManager(Generic[SceneState_contra]):
    """Observation manager that combines multiple observation managers into a dict.

    This manager evaluates multiple child observation managers and returns
    a dictionary mapping names to their individual observations. This is useful
    for combining multiple observation sources (e.g., state + camera + proprioception).

    The extras dictionary preserves the individual extras from each child manager.

    Type Parameters:
        SceneState_contra: The contravariant scene state type.

    Attributes:
        observation_managers: Mapping of names to observation manager instances.
    """

    def __init__(
        self,
        observation_managers: Mapping[str, ObservationManager[SceneState_contra, Any]],  # pyright: ignore[reportExplicitAny]
    ):
        """Initialize the composite observation manager.

        Args:
            observation_managers: A mapping from descriptive names to observation
                manager instances. Each manager will be evaluated on every step.
        """
        self.observation_managers = observation_managers

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """The Gymnasium observation space (a Dict space combining all child spaces).

        Returns:
            A gym.spaces.Dict containing the observation space for each child manager.
        """
        return gym.spaces.Dict(
            {
                name: manager.observation_space
                for name, manager in self.observation_managers.items()
            }
        )

    def compute(self, scene_state: SceneState_contra) -> tuple[ObsDict, Extras]:
        """Evaluate all child managers and return dict of observations.

        Args:
            scene_state: The current scene state to process.

        Returns:
            A tuple containing:
                - observations: Dict mapping manager names to their observations
                - extras: Dict mapping manager names to their individual extras
        """
        observations: ObsDict = {}
        extras: dict[str, Extras] = {}

        for manager_name, manager in self.observation_managers.items():
            obs, obs_extras = manager.compute(scene_state)  # pyright: ignore[reportAny]
            observations[manager_name] = obs
            extras[manager_name] = obs_extras

        return observations, extras
