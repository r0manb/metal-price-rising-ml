import json
import os
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import pickle
import tensorflow as tf

from utils.path_utils import get_absolute_path
from utils.datetime_utils import calculate_delta_time


class Predictor:
    TIMESTAMP_STRATEGIES = Literal["last", "median", "adaptive"]

    def __init__(
        self,
        path: Union[str, os.PathLike] = "",
        timestamp_strategy: TIMESTAMP_STRATEGIES = "adaptive",
    ) -> None:
        self.path = get_absolute_path(path)

        self.timestamp_strategies = {
            "last": self._last_timestamp,
            "median": self._median_timestamp,
            "adaptive": self._adaptive_timestamp,
        }
        self._validate_strategy(timestamp_strategy)
        self.timestamp_strategy = timestamp_strategy

        self._load_config()
        self._load_models()

    def _validate_strategy(self, strategy: str) -> None:
        if strategy not in self.timestamp_strategies:
            raise ValueError(f"Unknown timestamp strategy: {strategy}")

    def _last_timestamp(self, delta: ArrayLike) -> int:
        return delta[-1]

    def _median_timestamp(self, delta: ArrayLike) -> int:
        return int(np.median(delta))

    def _adaptive_timestamp(self, delta: ArrayLike) -> int:
        median = self._median_timestamp(delta)
        last_delta = self._last_timestamp(delta)
        if last_delta > 2 * median:
            return median
        return last_delta

    def _load_config(self) -> None:
        with open(self.path / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        self.window_size = config["window_size"]
        self.forecast_horizon = config["forecast_horizon"]

    def _load_models(self) -> None:
        self._model = tf.keras.models.load_model(self.path / "best_model.keras")
        with open(self.path / "best_scaler.pkl", "rb") as f:
            self._scaler = pickle.load(f)

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        if len(data) < self.window_size:
            raise ValueError(
                f"Not enough data for prediction: got {len(data)} rows, need {self.window_size}."
            )

        delta_time = calculate_delta_time(data.index)
        close = data["close"].values

        features = np.column_stack((close, delta_time))[-self.window_size :]
        features_scaled = self._scaler.transform(features)

        return features_scaled.reshape(1, self.window_size, 2)

    def get_future_timestamps(
        self,
        dates: Union[np.ndarray, pd.DatetimeIndex],
        timestamp_strategy: Optional[TIMESTAMP_STRATEGIES] = None,
    ) -> np.ndarray:
        strategy = timestamp_strategy if timestamp_strategy else self.timestamp_strategy
        self._validate_strategy(strategy)

        timestamps = dates.astype("datetime64[ns]")
        seconds = calculate_delta_time(timestamps)
        delta = self.timestamp_strategies[strategy](seconds)

        np_delta = np.timedelta64(delta, "s")
        future_steps = np.arange(1, self.forecast_horizon + 1) * np_delta

        return future_steps + timestamps[-1]

    def predict(
        self, data: Union[pd.DataFrame, np.ndarray], preprocessed: bool = False
    ) -> np.ndarray:
        if not preprocessed:
            data = self.preprocess_data(data)
        predicted_data = self._model.predict(data).reshape(-1, 1)

        dummy = np.zeros_like(predicted_data)
        response = self._scaler.inverse_transform(np.hstack((predicted_data, dummy)))

        return response[:, 0]

    def predict_with_dates(
        self,
        data: pd.DataFrame,
        timestamp_strategy: Optional[TIMESTAMP_STRATEGIES] = None,
        return_df: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], pd.DataFrame]:
        predicted_data = self.predict(data)
        dates = self.get_future_timestamps(data.index, timestamp_strategy)
        if return_df:
            return pd.DataFrame({"close": predicted_data}, index=dates)
        return predicted_data, dates
