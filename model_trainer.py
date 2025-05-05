import json
import os
import pathlib
from typing import Optional, Union

import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

from parsers.data_parser import DataParser


class ModelTrainer:
    def __init__(
        self,
        data_parser: DataParser,
        epochs: int = 100,
        train_percent: float = 0.8,
        val_percent: float = 0.1,
        window_size: int = 7,
        model_path: Union[str, os.PathLike] = "/",
    ):
        self.data_parser = data_parser
        self._scaler = MinMaxScaler((0, 1))

        self.epochs = epochs
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.window_size = window_size
        self.model_path = pathlib.Path(model_path)

        self.train_dates = None
        self.train_X = None
        self.train_y = None
        self.val_dates = None
        self.val_X = None
        self.val_y = None
        self.test_dates = None
        self.test_X = None
        self.test_y = None

        self._load_data()
        self._calculate_splits()
        self._initialize_scaler()
        self._preprocess_data()
        self._prepare_data()

        self._initialize_model()

    def _load_data(self) -> None:
        self.data = self.data_parser.fetch_data()

    def _calculate_splits(self) -> None:
        total_lenght = len(self.data) - self.window_size
        self.train_len = int(total_lenght * self.train_percent)
        self.val_len = int(total_lenght * self.val_percent) + self.train_len

    def _initialize_scaler(self):
        train_data = self.data.iloc[: self.train_len + self.window_size]
        self._scaler.fit(train_data[["close"]].values)

    def _preprocess_data(self) -> None:
        prices = self._scaler.transform(self.data[["close"]].values)
        dates = self.data.index.values

        X = []
        y = []
        for i in range(len(self.data) - self.window_size):
            X.append(prices[i : i + self.window_size])
            y.append(prices[i + self.window_size])

        self._X = np.array(X).reshape(-1, self.window_size, 1).astype(np.float32)
        self._y = np.array(y).astype(np.float32)
        self._dates = dates[self.window_size :]

    def _prepare_data(self) -> None:
        self.train_X = self._X[: self.train_len]
        self.train_y = self._y[: self.train_len]
        self.train_dates = self._dates[: self.train_len]

        self.val_X = self._X[self.train_len : self.val_len]
        self.val_y = self._y[self.train_len : self.val_len]
        self.val_dates = self._dates[self.train_len : self.val_len]

        self.test_X = self._X[self.val_len :]
        self.test_y = self._y[self.val_len :]
        self.test_dates = self._dates[self.val_len :]

    def _initialize_model(self):
        model = Sequential(
            [
                LSTM(
                    128,
                    return_sequences=True,
                    input_shape=(self.window_size, 1),
                    kernel_initializer="he_normal",
                ),
                Dropout(0.2),
                LSTM(64, return_sequences=True, kernel_initializer="he_normal"),
                Dropout(0.2),
                LSTM(32, kernel_initializer="he_normal"),
                Dense(16, activation="relu", kernel_initializer="he_normal"),
                Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mean_absolute_error"])
        self._model = model

    def _save_models(self, path: pathlib.Path) -> None:
        self._model.save(path / "best_model.keras")
        with open(path / "best_scaler.pkl", "wb") as f:
            pickle.dump(self._scaler, f)

    def _save_config(self, path: pathlib.Path) -> None:
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump({"window_size": self.window_size}, f)

    def train(self) -> None:
        self._model.fit(
            self.train_X,
            self.train_y,
            validation_data=(self.val_X, self.val_y),
            epochs=self.epochs,
        )

    def save(self, path: Optional[Union[str, os.PathLike]] = None) -> None:
        path = pathlib.Path(path).resolve() if path is not None else self.model_path
        path.mkdir(parents=True, exist_ok=True)

        self._save_config(path)
        self._save_models(path)
