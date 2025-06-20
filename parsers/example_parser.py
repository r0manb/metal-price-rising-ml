import datetime as dt

import pandas as pd
import requests

from .data_parser import DataParser


class ExampleParser(DataParser):
    def __init__(
        self,
        ticker: str,
        interval: str,
        start: dt.datetime = None,
        end: dt.datetime = None,
    ):
        super().__init__(ticker, start, end)
        self.interval = interval

    def fetch_data(self):
        response = requests.get(
            "https://www.example.ru/api/trading/candles",
            {
                "start": self._start.isoformat(),
                "end": self._end.isoformat(),
                "ticker": self.ticker,
                "interval": self.interval,
            },
        )
        data = response.json()["payload"]["candles"]
        procesed_data = self._normalize_data(data)

        self._data = procesed_data
        return self.get_data()

    def _normalize_data(self, data):
        df = pd.DataFrame(data)

        processed_data = df[["c", "date"]].copy()
        processed_data.columns = ["close", "date"]

        processed_data["date"] = pd.to_datetime(processed_data["date"], unit="s")
        processed_data.set_index("date", inplace=True)

        return processed_data
