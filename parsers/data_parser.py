from abc import ABC, abstractmethod
import datetime as dt
from typing import Optional

import pandas as pd


class DataParser(ABC):
    DEFAULT_START_DATE = dt.datetime(2000, 1, 1, 1, 1, tzinfo=dt.timezone.utc)

    def __init__(
        self,
        ticker: str,
        start: dt.datetime = None,
        end: dt.datetime = None,
    ) -> None:
        self._data: Optional[pd.DataFrame] = None
        self.ticker = ticker
        self.update_time_period(start, end)

    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _normalize_data(self, data) -> pd.DataFrame:
        pass

    def get_data(self) -> Optional[pd.DataFrame]:
        return self._data

    def update_time_period(self, start: dt.datetime, end: dt.datetime) -> None:
        self.start = start if start is not None else self.DEFAULT_START_DATE
        self.end = end if end is not None else dt.datetime.now(dt.timezone.utc)
