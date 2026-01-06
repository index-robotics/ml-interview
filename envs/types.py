from typing import Any

import numpy as np
import numpy.typing as npt

type Reward = float
type Terminated = bool
type Truncated = bool
type Extras = dict[str, Any]  # pyright: ignore[reportExplicitAny]
type CameraObservation = npt.NDArray[np.uint8]  # Shape: (height, width, 3) RGB
type ObsDict = dict[str, Any]  # pyright: ignore[reportExplicitAny]
