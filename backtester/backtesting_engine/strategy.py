# Houses the trading logic. Given a new MarketEvent, emit SignalEvents.
from .events import SignalEvent


class Strategy:
    """Interface for pluggable trading strategies."""

    def on_bar(self, market_event):
        raise NotImplementedError


class BuyLowSellHighStrategy(Strategy):
    """Extremely naive example strategy to illustrate the callback signature."""

    def __init__(self, symbol: str, buy_below: float, sell_above: float, size: int = 1):
        self.symbol = symbol
        self.buy_below = buy_below
        self.sell_above = sell_above
        self.size = size
        self.current_position = 0

    def on_bar(self, market_event):
        price = market_event.price_row["Close"]
        signals = []

        if self.current_position == 0 and price <= self.buy_below:
            signals.append(SignalEvent(market_event.timestamp, self.symbol, "LONG", strength=self.size))
            self.current_position += self.size
        elif self.current_position > 0 and price >= self.sell_above:
            signals.append(SignalEvent(market_event.timestamp, self.symbol, "EXIT", strength=self.size))
            self.current_position = 0

        return signals
