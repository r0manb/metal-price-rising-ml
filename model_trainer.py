import json
import os
import pathlib
from typing import Dict, Tuple, Optional, Union

import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

from parsers.data_parser import DataParser
from utils.path_utils import get_absolute_path
from utils.datetime_utils import calculate_delta_time


class ModelTrainer:
    def __init__(
        self,
        data_parser: DataParser,
        epochs: int = 100,
        train_percent: float = 0.8,
        val_percent: float = 0.1,
        window_size: int = 7,
        forecast_horizon: int = 3,
        model_path: Union[str, os.PathLike] = "models",
    ):
        self._data_parser = data_parser
        self._scaler = MinMaxScaler((0, 1))

        self.epochs = epochs
        self._train_percent = train_percent
        self._val_percent = val_percent
        self._window_size = window_size
        self._forecast_horizon = forecast_horizon
        self._model_path = get_absolute_path(model_path)

        self._train_dates = None
        self._train_X = None
        self._train_y = None
        self._val_dates = None
        self._val_X = None
        self._val_y = None
        self._test_dates = None
        self._test_X = None
        self._test_y = None

        self._load_data()
        self._calculate_splits()
        self._initialize_scaler()
        self._preprocess_data()
        self._prepare_data()

        self._initialize_model()

    @property
    def train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._train_X, self._train_y, self._train_dates

    @property
    def val_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._val_X, self._val_y, self._val_dates

    @property
    def test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._test_X, self._test_y, self._test_dates

    def _load_data(self) -> None:
        self._data = self._data_parser.fetch_data()
        self._data["delta_time"] = calculate_delta_time(self._data.index.values)

    def _calculate_splits(self) -> None:
        self._total_len = (
            len(self._data) - self._window_size - self._forecast_horizon + 1
        )
        self._train_len = int(self._total_len * self._train_percent)
        self._val_len = int(self._total_len * self._val_percent) + self._train_len

    def _initialize_scaler(self):
        train_data = self._data.iloc[: self._train_len + self._window_size]
        self._scaler.fit(train_data.values)

    def _preprocess_data(self) -> None:
        features = self._scaler.transform(self._data.values)
        dates = self._data.index.values

        X = []
        y = []
        for i in range(self._total_len):
            X.append(features[i : i + self._window_size])
            y.append(
                features[
                    i
                    + self._window_size : i
                    + self._window_size
                    + self._forecast_horizon,
                    0,
                ]
            )

        self._X = np.array(X).astype(np.float32)
        self._y = np.array(y).astype(np.float32)
        self._dates = dates[self._window_size : self._total_len + self._window_size]

    def _prepare_data(self) -> None:
        self._train_X = self._X[: self._train_len]
        self._train_y = self._y[: self._train_len]
        self._train_dates = self._dates[: self._train_len]

        self._val_X = self._X[self._train_len : self._val_len]
        self._val_y = self._y[self._train_len : self._val_len]
        self._val_dates = self._dates[self._train_len : self._val_len]

        self._test_X = self._X[self._val_len :]
        self._test_y = self._y[self._val_len :]
        self._test_dates = self._dates[self._val_len :]

    def _initialize_model(self):
        model = Sequential(
            [
                LSTM(
                    128,
                    return_sequences=True,
                    input_shape=(self._window_size, 2),
                    kernel_initializer="he_normal",
                ),
                Dropout(0.2),
                LSTM(64, return_sequences=True, kernel_initializer="he_normal"),
                Dropout(0.2),
                LSTM(32, kernel_initializer="he_normal"),
                Dense(16, activation="relu", kernel_initializer="he_normal"),
                Dense(self._forecast_horizon),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mean_absolute_error"])
        self._model = model

    def _inverse_close(self, scaled_arr: np.ndarray) -> np.ndarray:
        scaled_arr = scaled_arr.reshape(-1, 1)
        comb = np.hstack((scaled_arr, np.zeros_like(scaled_arr)))
        return self._scaler.inverse_transform(comb)[:, 0]

    def _save_models(self, path: pathlib.Path) -> None:
        self._model.save(path / "best_model.keras")
        with open(path / "best_scaler.pkl", "wb") as f:
            pickle.dump(self._scaler, f)

    def _save_config(self, path: pathlib.Path) -> None:
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "window_size": self._window_size,
                    "forecast_horizon": self._forecast_horizon,
                },
                f,
            )

    def train(self) -> None:
        self._model.fit(
            self._train_X,
            self._train_y,
            validation_data=(self._val_X, self._val_y),
            epochs=self.epochs,
        )

    def evaluate(self) -> Dict[str, float]:
        y_pred_scaled = self._model.predict(self._test_X)
        y_pred = self._inverse_close(y_pred_scaled)

        y_true_scaled = self._test_y.reshape(-1, 1)
        y_true = self._inverse_close(y_true_scaled)

        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
        }

    def save(self, path: Optional[Union[str, os.PathLike]] = None) -> None:
        path = get_absolute_path(path) if path is not None else self._model_path
        path.mkdir(parents=True, exist_ok=True)

        self._save_config(path)
        self._save_models(path)
