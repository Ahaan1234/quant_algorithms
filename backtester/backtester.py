import pandas as pd

class SimpleBacktester:
    def __init__(self, initial_capital, data):
        """
        initial_capital: Float, starting cash
        data: Pandas DataFrame with 'Close' column and DateTime index
        """
        self.cash = initial_capital
        self.position = 0  # Number of shares held
        self.data = data
        self.equity_curve = [] # To track performance over time
        
    def run(self):
        print("--- Starting Backtest ---")
        
        # Iterate through the data row by row (simulating time passing)
        for date, row in self.data.iterrows():
            price = row['Close']
            
            # 1. Update Portfolio Value
            current_equity = self.cash + (self.position * price)
            self.equity_curve.append(current_equity)
            
            # 2. Strategy Logic (Example: Buy if price < 100, Sell if > 110)
            # In a real library, this would be a separate class
            if self.position == 0 and price < 100:
                self.buy(price, 10) # Buy 10 shares
            
            elif self.position > 0 and price > 110:
                self.sell(price, 10) # Sell 10 shares
                
        self.calculate_performance()

    def buy(self, price, quantity):
        cost = price * quantity
        if self.cash >= cost:
            self.cash -= cost
            self.position += quantity
            print(f"BOUGHT {quantity} at {price}")
        else:
            print("Insufficient funds")

    def sell(self, price, quantity):
        if self.position >= quantity:
            revenue = price * quantity
            self.cash += revenue
            self.position -= quantity
            print(f"SOLD {quantity} at {price}")
        else:
            print("Not enough shares to sell")

    def calculate_performance(self):
        initial = self.equity_curve[0]
        final = self.equity_curve[-1]
        pnl = final - initial
        return_pct = (pnl / initial) * 100
        
        print("--- Results ---")
        print(f"Starting Equity: ${initial}")
        print(f"Final Equity:    ${final:.2f}")
        print(f"Total Return:    {return_pct:.2f}%")

# --- usage ---

# Create dummy data for testing
data = pd.DataFrame({
    'Close': [105, 99, 98, 102, 112, 115, 108]
}, index=pd.date_range(start='2023-01-01', periods=7))

# Initialize and run
backtest = SimpleBacktester(initial_capital=10000, data=data)
backtest.run()