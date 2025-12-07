# 1. The Architecture Overview
A robust backtesting library usually consists of five core components that interact with each other.

Here is the breakdown of what you need to build:

- Data Handler: Loads historical data (CSV, Database, API) and feeds it to the system piece-by-piece to simulate a live market. Crucially, it must prevent look-ahead bias (peeking at tomorrow's price today).

- Strategy Interface: A template where you define your logic (e.g., "If the price goes up 2%, buy").

- Portfolio/Account Manager: Tracks your cash balance, current positions, and total equity. It handles the accounting.

- Execution Handler: Simulates the filling of orders. This is where you calculate commission fees and slippage (the difference between the price you wanted and the price you got).

- Performance Reporter: Calculates statistics like Total Return, Sharpe Ratio, and Maximum Drawdown after the test is done.

# 2. The Core Logic (The Event Loop)
The heart of your library is the "Event Loop." It iterates through your historical data one timestamp at a time.

## The Loop Flow:

- Update: The Data Handler serves the next price bar.

- Evaluate: The Strategy looks at the new price and decides if it wants to Buy, Sell, or Hold.

- Execute: If the Strategy sends an order, the Execution Handler checks if you have enough cash and "fills" the order.

- Record: The Portfolio updates the account value based on the new price.

- Repeat: Move to the next timestamp.