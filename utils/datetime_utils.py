from typing import Union

import numpy as np
import pandas as pd


def calculate_delta_time(dates: Union[np.ndarray, pd.DatetimeIndex]) -> np.ndarray:
    timesteps = dates.astype(np.int64)
    return np.diff(timesteps, prepend=timesteps[0]) // 10**9
