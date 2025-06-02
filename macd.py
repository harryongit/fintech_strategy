import numpy as np
import pandas as pd
import talib as ta_lib
from concurrent.futures import ThreadPoolExecutor
import os
import time
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


# Define stock symbols and CSV file paths

csv_files = {
    'ITC': '/home/ubuntu/Desktop/stock/stockmkt/data/1day/day/ITC.csv',
    'TATAMTR': '/home/ubuntu/Desktop/stock/stockmkt/data/1day/day/TATAMTRDVR.csv',
    'RELIANCE': '/home/ubuntu/Desktop/stock/stockmkt/data/1day/day/RELIANCE.csv',
    'WIPRO': '/home/ubuntu/Desktop/stock/stockmkt/data/1day/day/WIPRO.csv'
}

def get_stock_data(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    return data

def calculate_macd(data):
    data['MACD'], data['Signal_Line'], data['MACD_Hist'] = ta_lib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return data

def generate_signals(data):
    data['Buy_Signal'] = (data['MACD'] > data['Signal_Line']) & (data['MACD'].shift(1) <= data['Signal_Line'].shift(1))
    data['Sell_Signal'] = (data['MACD'] < data['Signal_Line']) & (data['MACD'].shift(1) >= data['Signal_Line'].shift(1))
    return data

def assign_positions(data):
    data['Position'] = 0
    data.loc[data['Buy_Signal'], 'Position'] = 1
    data.loc[data['Sell_Signal'], 'Position'] = -1
    data['Position'] = data['Position'].replace(to_replace=0, method='ffill')
    return data

def calculate_returns(data):
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Market_Return'] * data['Position'].shift(1)
    data['Cumulative_Market_Return'] = (1 + data['Market_Return']).cumprod()
    data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod()
    return data

def format_backtest_output(output):
    keys_to_display = [
        'Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]', 
        'Equity Peak [$]', 'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]', 
        'Volatility (Ann.) [%]', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 
        'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration', 
        'Avg. Drawdown Duration', '# Trades', 'Win Rate [%]', 'Best Trade [%]', 
        'Worst Trade [%]', 'Avg. Trade [%]', 'Max. Trade Duration', 
        'Avg. Trade Duration', 'Profit Factor', 'Expectancy [%]', 'SQN', '_strategy'
    ]
    return "\n".join([f"{key}: {output[key]}" for key in keys_to_display if key in output])

class EnhancedMACDStrategy(Strategy):
    # Parameters for the strategy
    stop_loss_pct = 0.02  # Stop loss percentage
    trailing_stop_pct = 0.01  # Trailing stop percentage
    break_even_pct = 0.05  # Break-even percentage
    target_profit_pct = 0.1  # Target profit percentage
    risk_reward_ratio = target_profit_pct / stop_loss_pct  # Risk-Reward Ratio

    def init(self):
        self.macd = self.I(ta_lib.MACD, self.data.Close, 12, 26, 9)
        self.entry_price = None
        self.stop_loss_price = None
        self.target_price = None
        self.break_even_price = None

    def next(self):
        # Handle Buy Orders
        if crossover(self.macd[0], self.macd[1]) and not self.position:
            self.entry_price = self.data.Close[-1]
            self.stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
            self.target_price = self.entry_price * (1 + self.target_profit_pct)
            self.break_even_price = self.entry_price * (1 + self.break_even_pct)
            self.buy()
            print(f"Buy Order: Entry Price = {self.entry_price:.2f}, Stop Loss = {self.stop_loss_price:.2f}, Target = {self.target_price:.2f}")

        # Handle Trailing Stop and Target Profit
        if self.position.is_long:
            current_price = self.data.Close[-1]
            if current_price > self.entry_price:
                self.stop_loss_price = max(self.stop_loss_price, current_price * (1 - self.trailing_stop_pct))
                if current_price >= self.break_even_price:
                    self.stop_loss_price = max(self.stop_loss_price, self.entry_price)
                if current_price >= self.target_price:
                    self.position.close()
                    print(f"Sell Order: Target Price Reached = {current_price:.2f}")
        
        # Handle Stop Loss
        if self.position.is_long and self.data.Close[-1] <= self.stop_loss_price:
            self.position.close()
            print(f"Sell Order: Stop Loss Triggered = {self.data.Close[-1]:.2f}")

def process_stock(stock_name, csv_file_path):
    try:
        data = get_stock_data(csv_file_path)
        data = calculate_macd(data)
        data = generate_signals(data)
        data = assign_positions(data)
        data = calculate_returns(data)

        total_trades = data[data['Position'] != 0].shape[0]
        wins = data[data['Strategy_Return'] > 0].shape[0]
        winning_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        result = []
        result.append(f"\nProcessing {stock_name} - {os.path.basename(csv_file_path)}")
        startdate = data.index.min().strftime('%Y-%m-%d')
        end_date = data.index.max().strftime('%Y-%m-%d')
        result.append(f"{stock_name} Start Date: {startdate}")
        result.append(f"{stock_name} End Date: {end_date}")
        result.append(f"{stock_name} Total Trades: {total_trades}")
        result.append(f"{stock_name} Winning Trades: {wins}")
        result.append(f"{stock_name} Winning Rate: {winning_rate:.2f}%")
        result.append(f"{stock_name} Risk-Reward Ratio: {EnhancedMACDStrategy.risk_reward_ratio:.2f}")

        signals = data[data['Buy_Signal'] | data['Sell_Signal']]
        if not signals.empty:
            result.append(f"\n{stock_name} Signals Detected:")
            result.append(signals[['Close', 'MACD', 'Signal_Line', 'Buy_Signal', 'Sell_Signal']].to_string())
        else:
            result.append(f"No buy or sell signals detected for {stock_name}.")
        
        bt = Backtest(data, EnhancedMACDStrategy, cash=10000, commission=.002, exclusive_orders=True)
        output = bt.run()
        result.append(f"\n{stock_name} Backtest Summary:\n{format_backtest_output(output)}")
        return result
    except Exception as e:
        return [f"Error processing {stock_name}: {e}"]

def main():
    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_stock, stock_name, csv_file_path) for stock_name, csv_file_path in csv_files.items()]
        
        for future in futures:
            results.append(future.result())

    for result in results:
        for line in result:
            print(line)

    end_time = time.time()
    print(f"\nTotal Processing Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
