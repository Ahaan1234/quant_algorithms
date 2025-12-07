# Basic event objects passed around the engine loop.
# Each component talks to others through these lightweight data containers.

class Event:
    def __init__(self, event_type):
        self.type = event_type


class MarketEvent(Event):
    def __init__(self, timestamp, price_row):
        super().__init__("MARKET")
        self.timestamp = timestamp
        self.price_row = price_row


class SignalEvent(Event):
    def __init__(self, timestamp, symbol, signal_type, strength=1.0):
        super().__init__("SIGNAL")
        self.timestamp = timestamp
        self.symbol = symbol
        self.signal_type = signal_type  # e.g., "LONG" or "EXIT"
        self.strength = strength


class OrderEvent(Event):
    def __init__(self, timestamp, symbol, order_type, quantity, direction):
        super().__init__("ORDER")
        self.timestamp = timestamp
        self.symbol = symbol
        self.order_type = order_type  # e.g., "MKT" or "LMT"
        self.quantity = quantity
        self.direction = direction  # "BUY" or "SELL"


class FillEvent(Event):
    def __init__(self, timestamp, symbol, quantity, direction, fill_price, commission=0.0, slippage=0.0):
        super().__init__("FILL")
        self.timestamp = timestamp
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction
        self.fill_price = fill_price
        self.commission = commission
        self.slippage = slippage
