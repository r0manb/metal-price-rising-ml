import json
import os
import pathlib
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
import pickle
import tensorflow as tf


class Predictor:
    def __init__(self, path: Union[str, os.PathLike] = "/") -> None:
        self.path = pathlib.Path(path).resolve()

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

    def _preprocess_data(self, data: ArrayLike) -> np.ndarray:
        data = np.array(data).reshape(-1, 1)
        data_scaled = self._scaler.transform(data)

        return data_scaled.reshape(-1, self.window_size, 1)

    def predict(self, data: ArrayLike) -> np.ndarray:
        preprocessed_data = self._preprocess_data(data)
        predict_data = self._model.predict(preprocessed_data)
        response = self._scaler.inverse_transform(predict_data)

        return response.flatten()
