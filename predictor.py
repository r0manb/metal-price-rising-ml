import json
import os
from typing import Union

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from utils.path_utils import get_absolute_path
from utils.datetime_utils import calculate_delta_time


class Predictor:
    def __init__(self, path: Union[str, os.PathLike] = "") -> None:
        self.path = get_absolute_path(path)

        self._load_config()
        self._load_models()

    def _load_config(self) -> None:
        with open(self.path / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        self.window_size = config["window_size"]

    def _load_models(self) -> None:
        self._model = tf.keras.models.load_model(self.path / "best_model.keras")
        with open(self.path / "best_scaler.pkl", "rb") as f:
            self._scaler = pickle.load(f)

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        if len(data) < self.window_size:
            raise ValueError(
                f"Not enough data for prediction: got {len(data)} rows, need {self.window_size}."
            )

        delta_time = calculate_delta_time(data.index.values)
        close = data["close"].values

        features = np.column_stack((close, delta_time))[-self.window_size :]
        features_scaled = self._scaler.transform(features)

        return features_scaled.reshape(1, self.window_size, 2)

    def predict(
        self, data: Union[pd.DataFrame, np.ndarray], preprocessed: bool = False
    ) -> np.ndarray:
        if not preprocessed:
            data = self.preprocess_data(data)
        predict_data = self._model.predict(data)

        dummy = np.zeros_like(predict_data)
        response = self._scaler.inverse_transform(np.hstack((predict_data, dummy)))

        return response[:, 0]
