import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np

# Define the file paths for CSV data of different stocks
csv_files = {
    'VGUARD': '/path/to/VGUARD.csv',
    'RELIANCE': '/path/to/RELIANCE.csv',
    'AXISBANK': '/path/to/AXISBANK.csv',
}


# Define parameters for backtesting
cash = 100000  # Initial capital for the backtest
commission = 0.002  # Commission rate for each trade
tp = 10  # Take profit distance in ticks
sl = 10  # Stop loss distance in ticks
maxidf = 5  # Max intraday filled orders

class GreedyStrategy(Strategy):
    """
    Greedy strategy implementation based on the Pine Script logic.
    """
    def init(self):
        self.buy_signals = 0
        self.sell_signals = 0
        self.max_intra_day_orders = maxidf
        self.orders_filled_today = 0

    def next(self):
        """
        Define buy/sell logic based on strategy.
        """
        if self.orders_filled_today >= self.max_intra_day_orders:
            return  # Stop trading if max intraday orders are filled
        
        # Gap conditions
        upGap = self.data.Open[-1] > self.data.High[-2]
        dnGap = self.data.Open[-1] < self.data.Low[-2]
        
        # Position conditions
        dn = self.position.size < 0 and self.data.Open[-1] > self.data.Close[-1]
        up = self.position.size > 0 and self.data.Open[-1] < self.data.Close[-1]
        
        # Gap Up Entry
        if upGap:
            self.buy(slippage=1, stop=self.data.High[-2])
            self.orders_filled_today += 1
        elif dnGap:
            self.sell(slippage=1, stop=self.data.Low[-2])
            self.orders_filled_today += 1
        
        # Downtrend Condition
        if dn:
            self.sell(slippage=1, stop=self.data.Close[-1])
            self.orders_filled_today += 1
        
        # Uptrend Condition
        if up:
            self.buy(slippage=1, stop=self.data.Close[-1])
            self.orders_filled_today += 1

        # TP and SL logic
        if self.position.size != 0:
            XQty = abs(self.position.size)
            direction = -1 if self.position.size < 0 else 1
            
            # Take profit and stop loss levels
            lmP = self.position.avg_price + direction * tp * self.data.mintick
            slP = self.position.avg_price - direction * sl * self.data.mintick
            
            # Set the orders for TP and SL
            if self.orders_filled_today > 0:
                self.order(
                    "TP", 
                    size=XQty, 
                    price=lmP, 
                    oca_group="TPSL", 
                    reduce=True
                )
                self.order(
                    "SL", 
                    size=XQty, 
                    price=slP, 
                    oca_group="TPSL", 
                    reduce=True
                )
        
        # Cancel TP/SL orders if no longer needed
        if self.position.size == 0:
            self.cancel("TP")
            self.cancel("SL")
            
    def on_order_filled(self, order):
        """
        Update the filled order count when an order is filled.
        """
        if order.is_buy():
            self.buy_signals += 1
        elif order.is_sell():
            self.sell_signals += 1

### BACKTESTING FUNCTION ###
def process_backtest_results(stock_name, strategy):
    """
    Process and display results of backtesting for a stock.
    :param stock_name: Name of the stock.
    :param strategy: Strategy instance after backtest.
    """
    print(f"{stock_name}: Buy Signals - {strategy.buy_signals}, Sell Signals - {strategy.sell_signals}")

def process_stock(stock_name, file_path):
    """
    Run backtest for a single stock.
    :param stock_name: Name of the stock.
    :param file_path: Path to the stock data CSV.
    :return: Backtest results.
    """
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')  # Parse date column
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df = df.drop(columns=['Volume'], errors='ignore')  # Drop Volume column if it exists
    df = df[df['Close'] != 0]  # Remove rows with zero Close prices

    if df.empty:
        print(f"Error: Data for {stock_name} is empty.")
        return None
    
    try:
        bt = Backtest(df, GreedyStrategy, cash=cash, commission=commission)
        results = bt.run()
        strategy = bt._results._strategy  # Extract strategy instance for signal counts
    except Exception as e:
        print(f"Backtest failed for {stock_name}: {str(e)}")
        return None

    process_backtest_results(stock_name, strategy)
    return results

def main():
    """
    Main function to process all stocks concurrently.
    """
    for stock_name, file_path in csv_files.items():
        process_stock(stock_name, file_path)

if __name__ == "__main__":
    main()
