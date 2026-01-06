import math
from typing import Any, Generic, TypeVar, cast

import gymnasium as gym
import mujoco
import numpy as np
import numpy.typing as npt
from typing_extensions import final

from envs.types import Extras, Reward, Terminated, Truncated

from .managers import (
    ActionManager,
    ObservationManager,
    RewardManager,
    SceneManager,
    TerminationManager,
)

# MJState is always mujoco.mjtState.mjSTATE_FULLPHYSICS
type MJState = npt.NDArray[np.float64]
type MJCtrl = npt.NDArray[np.float64]


def raw_observation_space(model: mujoco.MjModel) -> gym.spaces.Box:
    """Create a Gymnasium Box space for the raw MJState observation.

    The MJState is the full physics state from MuJoCo (mjSTATE_FULLPHYSICS),
    which is a 1D float64 array containing positions, velocities, and other
    simulation state.

    Args:
        model: The MuJoCo model to create the observation space for.

    Returns:
        A Box space with shape (state_size,) and dtype float64, with unbounded
        values (-inf, inf).
    """
    state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float64
    )


def raw_action_space(model: mujoco.MjModel) -> gym.spaces.Box:
    """Create a Gymnasium Box space for the raw MJCtrl action.

    The MJCtrl is the control vector for MuJoCo, which is a 1D float64 array
    containing actuator controls.

    Args:
        model: The MuJoCo model to create the action space for.

    Returns:
        A Box space with shape (nu,) and dtype float64, bounded by the
        actuator control ranges defined in the model.
    """
    nu = model.nu
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]

    return gym.spaces.Box(low=low, high=high, shape=(nu,), dtype=np.float64)


def _remove_empty_subdicts(d: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    cleaned = {
        k: (_remove_empty_subdicts(v) if isinstance(v, dict) else v)  # pyright: ignore[reportUnknownArgumentType]
        for k, v in d.items()  # pyright: ignore[reportAny]
    }
    # Only remove empty dicts, not falsy values like 0.0, False, [], etc.
    filtered = {k: v for k, v in cleaned.items() if not (isinstance(v, dict) and not v)}  # pyright: ignore[reportUnknownVariableType]
    return cast(dict[str, Any], filtered)  # pyright: ignore[reportExplicitAny]


Observation_co = TypeVar("Observation_co", covariant=True)
Action_contra = TypeVar("Action_contra", contravariant=True)


@final
class MJManagerBasedEnv(Generic[Observation_co, Action_contra]):
    """Manager-based reinforcement learning environment for MuJoCo simulations.

    This environment class composes five manager types to create a complete RL
    environment: SceneManager (physics), ActionManager (action processing),
    ObservationManager (state processing), RewardManager (reward computation),
    and TerminationManager (episode ending logic).

    The environment uses MJState (MuJoCo full physics state) as the scene state
    and MJCtrl (control vector) as the scene action, but allows arbitrary
    observation and action types through the managers.

    Type Parameters:
        Observation_co: The covariant observation type returned to the agent.
        Action_contra: The contravariant action type received from the agent.

    Attributes:
        scene_manager: Manages MuJoCo physics simulation.
        observation_manager: Transforms MJState to agent observations.
        action_manager: Transforms agent actions to MJCtrl.
        reward_manager: Computes rewards from MJState.
        termination_manager: Determines episode termination/truncation.
    """

    def __init__(
        self,
        scene_manager: SceneManager[MJState, MJCtrl],
        observation_manager: ObservationManager[MJState, Observation_co],
        action_manager: ActionManager[Action_contra, MJCtrl],
        reward_manager: RewardManager[MJState],
        termination_manager: TerminationManager[MJState],
    ) -> None:
        """Initialize the MuJoCo manager-based environment.

        Args:
            scene_manager: Manager for MuJoCo physics simulation.
            observation_manager: Manager for processing observations.
            action_manager: Manager for processing actions.
            reward_manager: Manager for computing rewards.
            termination_manager: Manager for determining episode endings.
        """
        self.scene_manager = scene_manager
        self.observation_manager = observation_manager
        self.action_manager = action_manager
        self.reward_manager = reward_manager
        self.termination_manager = termination_manager

    @property
    def control_period(self) -> float:
        """Time between control updates in seconds.

        Bubbles up from the scene manager's control period.
        """
        return self.scene_manager.control_period  # type: ignore[reportAttributeAccessIssue]

    @property
    def observation_space(self) -> gym.spaces.Space[Observation_co]:
        """The Gymnasium observation space for this environment.

        Bubbles up from the observation manager's observation space.

        Returns:
            A Gymnasium Space object describing the structure and bounds of
            observations produced by this environment.
        """
        return self.observation_manager.observation_space

    @property
    def action_space(self) -> gym.spaces.Space[Action_contra]:
        """The Gymnasium action space for this environment.

        Bubbles up from the action manager's action space.

        Returns:
            A Gymnasium Space object describing the structure and bounds of
            actions consumed by this environment.
        """
        return self.action_manager.action_space

    def reset(self, *, rng: np.random.Generator) -> tuple[Observation_co, Extras]:
        """Reset the environment to its initial state.

        Returns:
            A tuple containing:
                - The initial observation
                - extras: Nested dict with 'scene' and 'observation' subdicts
                    containing debug info from respective managers
        """
        # TODO: maybe add a state to reset to?
        initial_state, scene_extras = self.scene_manager.reset(rng=rng)
        obs, obs_extras = self.observation_manager.compute(initial_state)

        # Compose subdicts.
        extras = _remove_empty_subdicts(
            {"scene": scene_extras, "observation": obs_extras}
        )
        return obs, extras

    def step(
        self, action: Action_contra
    ) -> tuple[Observation_co, Reward, Terminated, Truncated, Extras]:
        """Execute one environment step with the given action.

        Args:
            action: The action to execute in the environment.

        Returns:
            A tuple containing:
                - observation: The observation after the step
                - reward: The scalar reward for this step
                - terminated: True if episode ended naturally
                - truncated: True if episode ended due to time limit
                - extras: Nested dict with 'scene', 'observation', 'action',
                    'reward', and 'termination' subdicts containing debug info
                    from respective managers
        """
        sim_action, action_extras = self.action_manager.compute(action)
        sim_state, state_extras = self.scene_manager.step(sim_action)
        obs, obs_extras = self.observation_manager.compute(sim_state)
        reward, reward_extras = self.reward_manager.compute(sim_state)
        term, trunc, term_extras = self.termination_manager.compute(sim_state, reward)

        # Compose extras subdicts.
        extras = _remove_empty_subdicts(
            {
                "scene": state_extras,
                "observation": obs_extras,
                "action": action_extras,
                "reward": reward_extras,
                "termination": term_extras,
            }
        )
        return obs, reward, term, trunc, extras


##################################################################
# Specific managers.
##################################################################


def mjData_from_mjState(model: mujoco.MjModel, state: MJState) -> mujoco.MjData:
    """Create a fresh MjData and set its state from an MJState.

    The provided `state` is assumed to correspond to mujoco.mjtState.mjSTATE_FULLPHYSICS.
    This helper allocates a new MjData for `model`, applies the state, runs
    mj_forward to update derived quantities, and returns the MjData.
    """
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model, data)
    return data


class MJSceneManager:
    """Scene manager for MuJoCo physics simulations.

    This manager maintains a MuJoCo model and data instance, handling
    simulation stepping and state extraction. It represents the scene state
    as MJState (full physics state) and accepts MJCtrl (control vector) actions.

    The manager uses keyframe 0 from the model XML as the reset state.

    Attributes:
        model: The MuJoCo model defining the scene.
        data: The MuJoCo data instance holding the current simulation state.
        nstate: Size of the full physics state vector.
        decimation: Number of simulation steps per environment step (action repeat).
        control_period: Time between control updates in seconds.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    nstate: int
    decimation: int

    def __init__(self, model: mujoco.MjModel, decimation: int = 1):
        """Initialize the MuJoCo scene manager.

        Args:
            model: The MuJoCo model to simulate.
            decimation: Number of simulation steps per environment step (action repeat).
                Default is 1 (no decimation).
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.decimation = decimation

    @property
    def control_period(self) -> float:
        """Time between control updates in seconds (decimation * timestep)."""
        return self.decimation * self.model.opt.timestep

    def step(self, scene_action: MJCtrl) -> tuple[MJState, Extras]:
        """Advance the MuJoCo simulation by decimation timesteps.

        Args:
            scene_action: Control vector to apply to the simulation.

        Returns:
            A tuple containing:
                - The full physics state after stepping
                - An empty extras dictionary
        """
        # Step.
        if len(scene_action) != self.model.nu:
            raise ValueError(
                f"Expected control size {self.model.nu}, got {len(scene_action)}"
            )
        self.data.ctrl[:] = scene_action

        # Step decimation times (action repeat)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        # Extract the state.
        state = np.empty((self.nstate,))
        mujoco.mj_getState(
            self.model, self.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        return state, {}

    def reset(self, *, rng: np.random.Generator) -> tuple[MJState, Extras]:  # pyright: ignore[reportUnusedParameter]
        """Reset the simulation to keyframe 0 from the model XML.

        Returns:
            A tuple containing:
                - The full physics state after reset
                - An empty extras dictionary
        """
        mujoco.mj_resetData(self.model, self.data)
        # Reset to keyframe 0, which includes the initial positions from the XML
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        # Extract!
        state = np.empty((self.nstate,))
        mujoco.mj_getState(
            self.model, self.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        return state, {}


@final
class TimeoutTerminationManager:
    """Termination manager that truncates episodes after a time or step limit.

    This manager truncates (but does not terminate) episodes when they exceed
    a maximum number of simulation steps or simulation time. Exactly one of
    time_limit or max_steps must be provided.

    When time_limit is provided, it's converted to steps based on the model's
    timestep (rounded up).

    When max_steps is provided:
    - If fps is provided: max_steps represents environment steps and is converted
      to simulation steps: 50 (reset overhead) + n * num_steps_per_action
      where num_steps_per_action = (1/fps) / timestep
    - If fps is None: max_steps represents simulation steps directly

    Attributes:
        model: The MuJoCo model (used to extract time from state).
        max_steps: Maximum number of simulation steps before truncation.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        time_limit: float | None = None,
        max_steps: int | None = None,
        fps: float | None = None,
    ) -> None:
        """Initialize the timeout termination manager.

        Args:
            model: The MuJoCo model being simulated.
            time_limit: Maximum simulation time in seconds (mutually exclusive
                with max_steps). Will be converted to steps using model timestep.
            max_steps: Maximum number of environment steps (if fps is provided)
                or simulation steps (if fps is None).
            fps: Target frequency for actions in Hz. If provided, max_steps is
                interpreted as environment steps and converted to simulation steps.

        Raises:
            ValueError: If neither or both of time_limit/max_steps are provided,
                or if provided values are non-positive.
        """
        self.model = model
        match (max_steps, time_limit):
            case (None, None):
                raise ValueError("Either time_limit or max_steps must be provided.")
            case (ms, None):
                # Accept an integer.
                if ms <= 0:
                    raise ValueError("max_steps must be positive.")
                # Convert environment steps to simulation steps if fps is provided
                if fps is not None:
                    timestep = model.opt.timestep
                    action_interval = 1.0 / fps
                    num_steps_per_action = int(round(action_interval / timestep))
                    # 50 simulation steps for reset overhead + n * num_steps_per_action
                    self.max_steps = 50 + ms * num_steps_per_action
                else:
                    self.max_steps = ms
            case (None, tl):
                # Compute number of steps from time_limit and the model timestep.
                if tl <= 0:
                    raise ValueError("time_limit must be positive.")
                timestep = model.opt.timestep
                computed_ms = math.ceil(tl / timestep)
                # Ensure at least one step.
                self.max_steps = computed_ms
            case _:
                # Covers the case where both are provided (not None) or unexpected types.
                raise ValueError(
                    "Provide only one of time_limit or max_steps, not both."
                )

    def compute(
        self,
        scene_state: MJState,
        reward: Reward,  # pyright: ignore[reportUnusedParameter]
    ) -> tuple[Terminated, Truncated, Extras]:
        """Check if the episode has exceeded the step/time limit.

        Args:
            scene_state: The current MuJoCo state.
            reward: The current reward (unused).

        Returns:
            A tuple containing:
                - terminated: Always False (timeouts are truncations)
                - truncated: True if step count >= max_steps
                - extras: Dict with 'termination_t' (simulation time) and
                    'termination_nsteps' (current step count)
        """
        # it's a BIT of a code smell to have to get the data object back...
        # TODO(ben): Figure out if we can pass around MJState freely? or not.
        data = mjData_from_mjState(self.model, scene_state)
        t = data.time
        termination_nsteps = int(t / self.model.opt.timestep)
        truncate = termination_nsteps >= self.max_steps
        return (
            False,
            truncate,
            {"termination_t": t, "termination_nsteps": termination_nsteps},
        )


@final
class ProximityRewardManager:
    """Reward manager that gives binary rewards based on distance to a goal.

    This manager computes the Euclidean distance from a MuJoCo site to a goal
    location and returns 1.0 if within threshold, 0.0 otherwise. This is useful
    for sparse reward reaching tasks.

    Attributes:
        model: The MuJoCo model (used to reconstruct data from state).
        site_name: Name of the MuJoCo site to track.
        goal_location: 3D target position in world coordinates.
        dist_threshold: Distance threshold for reward (reward=1 if dist<=threshold).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        # TODO(ben): Make this more generic not just to sites, but also to other things.
        site_name: str,
        goal_location: npt.NDArray[np.float64],
        dist_threshold: float,
    ) -> None:
        """Initialize the proximity reward manager.

        Args:
            model: The MuJoCo model being simulated.
            site_name: Name of the site to measure distance from (must exist in model).
            goal_location: Target 3D position [x, y, z] in world coordinates.
            dist_threshold: Maximum distance for positive reward.
        """
        self.model = model
        self.site_name = site_name
        self.goal_location = goal_location
        self.dist_threshold = dist_threshold

    def compute(self, scene_state: MJState) -> tuple[Reward, Extras]:
        """Compute binary proximity reward based on distance to goal.

        Args:
            scene_state: The current MuJoCo state.

        Returns:
            A tuple containing:
                - reward: 1.0 if distance <= threshold, 0.0 otherwise
                - extras: Dict with 'dist' containing the actual distance
        """
        data = mjData_from_mjState(self.model, scene_state)
        pos = np.asarray(data.site(self.site_name).xpos)  # pyright: ignore[reportAny]
        dist = np.linalg.norm(pos - self.goal_location)
        reward = 1.0 if dist <= self.dist_threshold else 0.0
        return reward, {"dist": dist}
