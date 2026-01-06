"""Trajectory planning utilities."""

import numpy as np
from numpy.typing import NDArray


class LinearJointTrajectory:
    """Linear joint space trajectory with per-joint durations."""

    def __init__(
        self,
        start_q: NDArray[np.float64],
        target_q: NDArray[np.float64],
        joint_durations: NDArray[np.float64],
    ) -> None:
        """Initialize linear joint trajectory.

        Args:
            start_q: Starting joint configuration
            target_q: Target joint configuration
            joint_durations: Duration for each joint to reach target (in seconds)
        """
        self.start_q = np.array(start_q, dtype=np.float64)
        self.target_q = np.array(target_q, dtype=np.float64)
        self.joint_durations = np.array(joint_durations, dtype=np.float64)
        self.duration = float(np.max(joint_durations))

    def __call__(self, t: float) -> NDArray[np.float64]:
        """Evaluate trajectory at time t.

        Args:
            t: Time in seconds (0 <= t <= duration)

        Returns:
            Joint configuration at time t
        """
        t = np.clip(t, 0.0, self.duration)
        q = self.start_q.copy()

        for i in range(len(self.start_q)):
            if t >= self.joint_durations[i]:
                q[i] = self.target_q[i]
            else:
                alpha = t / self.joint_durations[i]
                q[i] = (1 - alpha) * self.start_q[i] + alpha * self.target_q[i]

        return q

