import os
import pathlib
from typing import Optional, Union

import numpy as np
import pandas as pd
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
        self.scaler = MinMaxScaler((0, 1))

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
        self.scaler.fit(train_data[["close"]].values)

    def _preprocess_data(self) -> None:
        prices = self.scaler.transform(self.data[["close"]].values)
        dates = self.data.index.values

        X = []
        y = []
        window_dates = []
        for i in range(len(self.data) - self.window_size):
            X.append(prices[i : i + self.window_size])
            y.append(prices[i + self.window_size])
            window_dates.append(dates[i + self.window_size])

        windows = pd.DataFrame({})
        windows["date"] = window_dates

        X = np.array(X)
        for i in range(self.window_size):
            windows[f"day - {self.window_size - i}"] = X[:, i]
        windows["target"] = y

        self.preprocessed_data = windows

    def _prepare_data(self) -> None:
        data = self.preprocessed_data.to_numpy()

        X = data[:, 1:-1]
        X = X.reshape(*X.shape, 1).astype(np.float32)
        y = data[:, -1].astype(np.float32)
        dates = data[:, 0]

        self.train_X = X[:self.train_len]
        self.train_y = y[:self.train_len]
        self.train_dates = dates[:self.train_len]

        self.val_X = X[self.train_len:self.val_len]
        self.val_y = y[self.train_len:self.val_len]
        self.val_dates = dates[self.train_len:self.val_len]

        self.test_X = X[self.val_len:]
        self.test_y = y[self.val_len:]
        self.test_dates = dates[self.val_len:]

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
        self.model = model

    def train(self):
        self.model.fit(
            self.train_X,
            self.train_y,
            validation_data=(self.val_X, self.val_y),
            epochs=self.epochs,
        )

    def save(self, path: Optional[Union[str, os.PathLike]] = None):
        path = pathlib.Path(path) if path is not None else self.model_path
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(self.model_path / "best_model.keras")
        with open(self.model_path / "best_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
