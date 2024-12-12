import pandas as pd
from backtesting import Backtest, Strategy

# Define the file paths for CSV data of different stocks
csv_files = {
    'VGUARD': '/path/to/VGUARD.csv',
    'RELIANCE': '/path/to/RELIANCE.csv',
    'AXISBANK': '/path/to/AXISBANK.csv',
}

# Define parameters for backtesting
cash = 100000  # Initial capital for the backtest
commission = 0.002  # Commission rate for each trade

### BARUPDN FUNCTION ###
def barupdn(data):
    """
    Calculate the trend based on bar-up and bar-down logic.
    :param data: DataFrame with stock data.
    :return: 1 for uptrend (bar up), -1 for downtrend (bar down), 0 for neutral.
    """
    if data.Close[-1] > data.Close[-2]:
        return 1  # Uptrend (bar up)
    elif data.Close[-1] < data.Close[-2]:
        return -1  # Downtrend (bar down)
    return 0  # Neutral (no clear trend)

### STRATEGY CLASS ###
class BarupdnStrategy(Strategy):
    """
    Define the Barupdn strategy based purely on bar up/down logic.
    """
    def init(self):
        """
        Initialize the strategy (no indicators needed for this simple strategy).
        """
        self.buy_signals = 0
        self.sell_signals = 0

    def next(self):
        """
        Define buy/sell logic for each step in backtesting.
        """
        current_trend = barupdn(self.data)

        if current_trend == 1:
            self.buy()  # Buy if bar is up (bullish trend)
            self.buy_signals += 1
        elif current_trend == -1:
            self.sell()  # Sell if bar is down (bearish trend)
            self.sell_signals += 1

### BACKTEST PROCESSING ###
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
        bt = Backtest(df, BarupdnStrategy, cash=cash, commission=commission)
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
