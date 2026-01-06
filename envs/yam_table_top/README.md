# YAM Tabletop Manipulation Environment Suite

This module provides a MimicLabs-style environment suite for the YAM arm on a tabletop workspace.

## Installation

### As part of mechacarpal

If you're using this as part of the mechacarpal package, it's already available after installing mechacarpal:

```bash
# Install mechacarpal (from the repository root)
pip install -e .
```

### As a standalone package

To install this package separately (requires mechacarpal.core.envs to be available):

```bash
cd src/mechacarpal/core/envs/yam_table_top
pip install -e .
```

Or using pip directly:

```bash
pip install -e src/mechacarpal/core/envs/yam_table_top
```

**Note**: This package depends on `mechacarpal.core.envs` for base classes like `MJManagerBasedEnv`, `MJSceneManager`, etc. Make sure mechacarpal is installed first.

## Structure

The environment suite follows the MimicLabs organization pattern:

- **`scene.py`**: Scene/arena creation with table and YAM arm
- **`robot.py`**: YAM robot configuration and properties
- **`objects.py`**: Object definitions for manipulation tasks (Block, Sphere, Cylinder)
- **`env.py`**: Main environment class using the manager-based framework

## Usage

### Basic Usage

```python
import numpy as np
from mechacarpal.core.envs.yam_table_top import YAMTableTopEnv

# Create environment
env = YAMTableTopEnv(
    table_size=(0.6, 0.4, 0.02),
    max_steps=1000,
)

# Reset
rng = np.random.default_rng(42)
obs, extras = env.reset(rng=rng)

# Step
action = np.zeros(env.n_actuators)  # 7 DOF for YAM arm
obs, reward, terminated, truncated, extras = env.step(action)

# Get end-effector pose
eef_pose = env.get_robot_eef_pose()  # 4x4 transformation matrix
```

### Using the Rollout Script

A convenient rollout script is provided:

```bash
# Run with default settings (headless)
python src/mechacarpal/core/scripts/rollout_yam.py

# Run with offscreen rendering (video recording)
python src/mechacarpal/core/scripts/rollout_yam.py \
    --visualizer-type offscreen \
    --camera-type fixed \
    --output-dir ./videos

# Run with interactive viewer
python src/mechacarpal/core/scripts/rollout_yam.py \
    --visualizer-type viewer

# Customize table and goal
python src/mechacarpal/core/scripts/rollout_yam.py \
    --table-size 0.8 0.6 0.02 \
    --goal-location 0.2 -0.15 0.5 \
    --max-steps 10000
```

## Custom Managers

The environment uses the manager-based framework, allowing you to customize:

- **ObservationManager**: Transform scene state to observations
- **ActionManager**: Transform agent actions to control signals
- **RewardManager**: Compute task-specific rewards
- **TerminationManager**: Define episode termination conditions

Example with custom reward:

```python
from mechacarpal.core.envs.mj_env import ProximityRewardManager
from mechacarpal.core.envs.yam_table_top import YAMTableTopEnv, YAM_FINGERTIP_GRASP_SITE

# Create reward manager for reaching a goal
goal_location = np.array([0.2, 0.1, 0.5])
reward_manager = ProximityRewardManager(
    model=model,  # Will be created by env
    site_name=YAM_FINGERTIP_GRASP_SITE,
    goal_location=goal_location,
    dist_threshold=0.05,
)

env = YAMTableTopEnv(reward_manager=reward_manager)
```

## Objects

The `objects.py` module provides object classes for manipulation tasks:

- `Block`: Rectangular block
- `Sphere`: Spherical object
- `Cylinder`: Cylindrical object

These can be added to the scene using the `add_to_spec` method.

