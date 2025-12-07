import pandas as pd

from backtesting_engine import build_naive_example

# Create dummy data
data = pd.DataFrame(
    {"Close": [100, 105, 110, 95, 120]},
    index=pd.date_range(start="2023-01-01", periods=5),
)

# Quick smoke test: wires up the default components declared in the package.
engine = build_naive_example(initial_capital=5000, price_data=data)
engine.run()
