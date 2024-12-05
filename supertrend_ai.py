import pandas as pd
import numpy as np
import talib as ta
from backtesting import Backtest, Strategy
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.cluster import KMeans

# Define the file paths for CSV data of different stocks
csv_files = {
    'VGUARD': '/path/to/VGUARD.csv',
    'RELIANCE': '/path/to/RELIANCE.csv',
    'AXISBANK': '/path/to/AXISBANK.csv',
}

# Define parameters for backtesting
cash = 100000  # Initial capital for the backtest
commission = 0.002  # Commission rate for each trade
atr_length = 10  # ATR calculation period
min_mult = 1  # Minimum multiplier for SuperTrend
max_mult = 5  # Maximum multiplier for SuperTrend
step = 0.5  # Step size for SuperTrend multipliers

### UTILITY FUNCTIONS ###
def calculate_atr(data, period=14):
    """
    Calculate the Average True Range (ATR) using TA-Lib.
    :param data: DataFrame with High, Low, and Close columns.
    :param period: Period for ATR calculation.
    :return: ATR values.
    """
    return ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=period)

def calculate_supertrend(factor, atr, hl2):
    """
    Calculate SuperTrend upper and lower bands.
    :param factor: Multiplier for ATR.
    :param atr: ATR values.
    :param hl2: Average of High and Low prices.
    :return: Upper and lower bands of SuperTrend.
    """
    upper = hl2 + atr * factor
    lower = hl2 - atr * factor
    return upper, lower

def apply_kmeans_clustering(performance_data, num_clusters=3):
    """
    Apply KMeans clustering to performance data.
    :param performance_data: Matrix of performance data.
    :param num_clusters: Number of clusters.
    :return: Cluster labels and cluster centers.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(performance_data)
    return kmeans.labels_, kmeans.cluster_centers_

def calculate_supertrend_kmeans(data, atr, hl2, factors, alpha=0.1):
    """
    Calculate the optimal SuperTrend factor using KMeans clustering.
    :param data: Backtesting data object.
    :param atr: ATR values.
    :param hl2: Average of High and Low prices.
    :param factors: List of SuperTrend multipliers.
    :param alpha: Weighting factor for performance calculation.
    :return: Best-performing SuperTrend factor.
    """
    performances = []
    
    # Convert Backtesting data to pandas Series
    close_prices = pd.Series(data.Close, index=data.index)  # Convert Close prices
    hl2_series = pd.Series(hl2, index=data.index)  # Convert HL2 values
    
    for factor in factors:
        # Calculate SuperTrend bands
        upper, lower = calculate_supertrend(factor, atr, hl2_series)
        
        # Generate buy/sell signals based on crossover
        supertrend_signal = np.where(close_prices > upper, 1, np.where(close_prices < lower, -1, 0))
        
        # Calculate performance (price change weighted by signal)
        delta_price = close_prices.diff()  # Price change
        performance = alpha * (delta_price * pd.Series(supertrend_signal).shift(1))
        performances.append(performance)
    
    # Combine performances into a matrix for clustering
    performance_matrix = np.array(performances).T
    
    # Apply KMeans clustering
    labels, centers = apply_kmeans_clustering(performance_matrix)
    
    # Find the best-performing cluster and its factors
    best_cluster_idx = np.argmax(np.mean(centers, axis=1))
    best_factors = [factors[i] for i in range(len(factors)) if labels[i] == best_cluster_idx]
    
    return np.mean(best_factors)

def get_stock_data(file_path):
    """
    Load stock data from CSV and prepare it for backtesting.
    :param file_path: Path to the CSV file.
    :return: Prepared DataFrame.
    """
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')  # Parse date column
    data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    data = data.drop(columns=['Volume'], errors='ignore')  # Drop Volume column if it exists
    data = data[data['Close'] != 0]  # Remove rows with zero Close prices
    return data

### STRATEGY CLASS ###
class SupertrendStrategy(Strategy):
    """
    Define the trading strategy based on SuperTrend.
    """
    atr_length = atr_length
    min_mult = min_mult
    max_mult = max_mult
    step = step

    def init(self):
        """
        Initialize indicators and variables.
        """
        # Calculate ATR and HL2
        self.atr = self.I(ta.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_length)
        self.hl2 = (self.data.High + self.data.Low) / 2
        
        # Define factors for SuperTrend
        self.factors = np.arange(self.min_mult, self.max_mult + self.step, self.step)
        
        # Find the optimal SuperTrend factor
        self.best_factor = calculate_supertrend_kmeans(self.data, self.atr, self.hl2, self.factors)
        
        # Calculate SuperTrend bands
        self.upper, self.lower = calculate_supertrend(self.best_factor, self.atr, self.hl2)
        
        # Initialize signals and trend tracking
        self.buy_signals = 0
        self.sell_signals = 0
        self.previous_trend = 0

    def next(self):
        """
        Define buy/sell logic for each step in backtesting.
        """
        current_trend = np.nan  # Placeholder for current trend
        
        # Determine trend based on SuperTrend bands
        if self.data.Close[-1] > self.upper[-1]:
            current_trend = 1  # Uptrend
        elif self.data.Close[-1] < self.lower[-1]:
            current_trend = -1  # Downtrend
        
        # Execute buy/sell orders based on trend changes
        if current_trend == 1 and self.previous_trend <= 0:
            self.buy()
            self.buy_signals += 1
        elif current_trend == -1 and self.previous_trend >= 0:
            self.sell()
            self.sell_signals += 1
        
        self.previous_trend = current_trend  # Update trend

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
    df = get_stock_data(file_path)
    
    if df.empty:
        print(f"Error: Data for {stock_name} is empty.")
        return None
    
    try:
        bt = Backtest(df, SupertrendStrategy, cash=cash, commission=commission)
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
    start_time = time.time()
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_stock, stock, path) for stock, path in csv_files.items()]
        _ = [future.result() for future in futures if future.result() is not None]
    
    end_time = time.time()
    print(f"Total Time Taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
