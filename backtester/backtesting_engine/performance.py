# Records the account equity over time and produces summary stats.
import numpy as np
import pandas as pd


class PerformanceReporter:
    def __init__(self):
        self.equity_curve = []

    def record(self, timestamp, equity):
        self.equity_curve.append({"timestamp": timestamp, "equity": equity})

    def summary(self):
        curve = pd.DataFrame(self.equity_curve)
        if curve.empty:
            return {}

        returns = curve["equity"].pct_change().dropna()
        total_return = (curve["equity"].iloc[-1] / curve["equity"].iloc[0]) - 1
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)
        max_drawdown = self._max_drawdown(curve["equity"])

        return {
            "total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
        }

    @staticmethod
    def _max_drawdown(equity_series):
        cummax = equity_series.cummax()
        drawdowns = (equity_series - cummax) / cummax
        return drawdowns.min()
