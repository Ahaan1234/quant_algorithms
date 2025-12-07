"""The event loop that coordinates data, strategy, execution, and accounting."""
from .data_handler import DataFrameDataHandler
from .execution import SimulatedExecutionHandler
from .performance import PerformanceReporter
from .portfolio import NaivePortfolio
from .strategy import BuyLowSellHighStrategy


class SimpleBacktester:
    """
    A minimal orchestrator to show how the pieces talk to each other.
    In production you would swap in different handlers/strategies without
    touching this control flow.
    """

    def __init__(
        self,
        data_handler,
        strategy,
        portfolio,
        execution_handler,
        performance_reporter,
    ):
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.performance_reporter = performance_reporter

    def run(self):
        print("--- Starting Backtest ---")

        # Classic event loop: pull data -> generate signals -> create orders -> fill -> update PnL.
        while self.data_handler.has_data():
            market_event = self.data_handler.stream_next()
            if market_event is None:
                break

            # Strategy reacts to new bar and emits signals.
            signals = self.strategy.on_bar(market_event) or []

            # Portfolio turns signals into orders, which get filled by the execution handler.
            for signal in signals:
                order = self.portfolio.generate_order(signal)
                if order is None:
                    continue
                fill = self.execution_handler.execute_order(order, market_event)
                self.portfolio.update_fill(fill, market_event.price_row["Close"])

            # Record equity after processing this bar.
            equity = self.portfolio.equity(market_event.price_row["Close"])
            self.performance_reporter.record(market_event.timestamp, equity)

        results = self.performance_reporter.summary()
        self._print_results(results)

    @staticmethod
    def _print_results(results):
        print("--- Results ---")
        for metric, value in results.items():
            print(f"{metric}: {value}")


def build_naive_example(initial_capital, price_data):
    """
    Helper for quick prototyping: wires up all default handlers/strategy.
    Keeps test.py lightweight while showing the architecture.
    """
    data_handler = DataFrameDataHandler(price_data)
    strategy = BuyLowSellHighStrategy(symbol="ASSET", buy_below=100, sell_above=110, size=10)
    portfolio = NaivePortfolio(initial_cash=initial_capital, symbol="ASSET")
    execution_handler = SimulatedExecutionHandler(commission_per_trade=1.0, slippage_per_share=0.01)
    performance_reporter = PerformanceReporter()

    return SimpleBacktester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        performance_reporter=performance_reporter,
    )
