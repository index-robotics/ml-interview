# ml-interview

ML Interview environments - YAM tabletop manipulation environments built on MuJoCo.

## Installation

### From Source

Install the package in development mode:

```bash
# Clone the repository (with submodules)
git clone --recursive <repository-url>
cd ml-interview

# Or if you already cloned without submodules:
# git submodule update --init --recursive

# Install the package (i2rt will be automatically installed from the submodule)
pip install -e .
```

### Dependencies

The package requires:
- Python >= 3.12
- MuJoCo >= 3.3.7
- NumPy >= 1.24.0
- SciPy >= 1.16.3
- Gymnasium >= 0.29.0
- robot-descriptions >= 1.21.0
- trimesh >= 3.20.0
- robosuite >= 1.3.0

**i2rt dependency:**
- `i2rt` package is included as a git submodule under `third_party/i2rt` and will be automatically installed when installing this package
- Required for inverse kinematics when using `action_mode="pose"`
- If you only use joint-based actions (`action_mode="joints"`), i2rt is still installed but not actively used

## Importing the Package

After installation, you can import the package using the `envs` module name:

```python
# Import main components from the top-level package
from envs import YAMTableTopEnv, YAMPickAndPlaceBlockEnv

# Or import from submodules
from envs.yam_table_top import YAMTableTopEnv
from envs.yam_table_top.tasks import YAMPickAndPlaceBlockEnv

# Import utilities
from envs.common import rotation_6d_to_matrix, T_gripper
from envs.common.camera_config import FixedCamera, TrackingCamera
from envs.managers import ActionManager, ObservationManager
```

## Usage

### Basic Usage

```python
import numpy as np
from envs import YAMTableTopEnv

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
```

### Pick and Place Task

```python
from envs import YAMPickAndPlaceBlockEnv

# Create pick-and-place environment
env = YAMPickAndPlaceBlockEnv(
    max_steps=1000,
    randomize_block=True,  # Randomize block position on reset
)

# Use the environment
rng = np.random.default_rng(42)
obs, extras = env.reset(rng=rng)
```

### Using Submodules Directly

You can also import from specific submodules:

```python
# Import from yam_table_top submodule
from envs.yam_table_top.core import YAMTableTopEnv, YAMRobot
from envs.yam_table_top.core.objects import Block, Sphere, Cylinder

# Import managers and utilities
from envs.managers import RewardManager, TerminationManager
from envs.mj_env import MJSceneManager, MJState
from envs.common.robot_utils import rotation_6d_to_matrix
```

## Package Structure

- `envs/` - Main package directory
  - `mj_env.py` - Core MuJoCo environment base classes
  - `managers.py` - Manager protocols and implementations
  - `types.py` - Type definitions
  - `yam_table_top/` - YAM tabletop manipulation environments
    - `core/` - Core environment components
    - `tasks/` - Task-specific environments
  - `randomization/` - Physics randomization utilities
  - `common/` - Common utilities (camera config, geometry, interpolators)
  - `planning/` - Trajectory planning utilities
- `third_party/` - Third-party dependencies as git submodules
  - `i2rt/` - i2rt robotics library (for inverse kinematics)

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run linting:

```bash
ruff check .
```
