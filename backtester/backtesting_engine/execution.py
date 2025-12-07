# Simulates market/exchange interaction to turn orders into fills.
from .events import FillEvent


class ExecutionHandler:
    """Interface for sending orders to a broker or a simulator."""

    def execute_order(self, order_event, market_event):
        raise NotImplementedError


class SimulatedExecutionHandler(ExecutionHandler):
    """Instant fill at the current market price with simple costs."""

    def __init__(self, commission_per_trade: float = 0.0, slippage_per_share: float = 0.0):
        self.commission_per_trade = commission_per_trade
        self.slippage_per_share = slippage_per_share

    def execute_order(self, order_event, market_event):
        price = market_event.price_row["Close"]
        slippage = self.slippage_per_share * order_event.quantity
        return FillEvent(
            timestamp=market_event.timestamp,
            symbol=order_event.symbol,
            quantity=order_event.quantity,
            direction=order_event.direction,
            fill_price=price,
            commission=self.commission_per_trade,
            slippage=slippage,
        )
