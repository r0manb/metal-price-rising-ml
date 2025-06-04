import numpy as np


def calculate_delta_time(dates: np.ndarray) -> np.ndarray:
    timesteps = dates.astype(np.int64)
    return np.diff(timesteps, prepend=timesteps[0]) // 10**9
