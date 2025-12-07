# Feeds market data into the engine one bar at a time to avoid look-ahead bias.
import pandas as pd

from .events import MarketEvent


class DataHandler:
    """Interface for streaming historical data into the engine."""

    def has_data(self):
        raise NotImplementedError

    def stream_next(self):
        raise NotImplementedError


class DataFrameDataHandler(DataHandler):
    """Simple DataFrame-backed handler for quick experiments."""

    def __init__(self, price_data: pd.DataFrame, symbol: str = "ASSET"):
        self.data = price_data
        self.symbol = symbol
        self._iterator = self.data.iterrows()
        self._finished = False

    def has_data(self):
        # Loop ends once the iterator is exhausted.
        return not self._finished

    def stream_next(self):
        try:
            timestamp, row = next(self._iterator)
        except StopIteration:
            self._finished = True
            return None
        return MarketEvent(timestamp=timestamp, price_row=row)
