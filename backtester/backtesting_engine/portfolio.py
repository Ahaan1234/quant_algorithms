# Tracks positions/cash and turns signals into executable orders.
from .events import OrderEvent, FillEvent


class Portfolio:
    """Interface defining how a portfolio reacts to signals and fills."""

    def generate_order(self, signal_event):
        raise NotImplementedError

    def update_fill(self, fill_event, latest_price):
        raise NotImplementedError

    def equity(self, latest_price):
        raise NotImplementedError


class NaivePortfolio(Portfolio):
    """Minimal cash/position tracker for a single symbol."""

    def __init__(self, initial_cash: float, symbol: str):
        self.cash = initial_cash
        self.symbol = symbol
        self.position = 0

    def generate_order(self, signal_event):
        if signal_event.signal_type == "LONG":
            return OrderEvent(
                timestamp=signal_event.timestamp,
                symbol=self.symbol,
                order_type="MKT",
                quantity=int(signal_event.strength),
                direction="BUY",
            )
        if signal_event.signal_type == "EXIT":
            return OrderEvent(
                timestamp=signal_event.timestamp,
                symbol=self.symbol,
                order_type="MKT",
                quantity=int(signal_event.strength),
                direction="SELL",
            )
        return None

    def update_fill(self, fill_event, latest_price):
        signed_qty = fill_event.quantity if fill_event.direction == "BUY" else -fill_event.quantity
        cost = fill_event.fill_price * fill_event.quantity
        total_cost = cost + fill_event.commission + fill_event.slippage

        if fill_event.direction == "BUY":
            self.cash -= total_cost
        else:
            self.cash += total_cost

        self.position += signed_qty

    def equity(self, latest_price):
        return self.cash + self.position * latest_price
