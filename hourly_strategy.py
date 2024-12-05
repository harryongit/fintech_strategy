import numpy as np
import pandas as pd
import talib as ta_lib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Paths to CSV files
csv_files = {
    'VGUARD': '/home/ubuntu/Desktop/stock/stockmkt/data/3minute/VGUARD.csv',
    'WIPRO': '/home/ubuntu/Desktop/stock/stockmkt/data/60minute/60minute/WIPRO.csv',
}

# Trading parameters
stop_loss = 5
target_profit = 10
cash = 10000
commission = 0.0025
exclusive_orders = True
zone_width = 0.01

def get_stock_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    required_columns = ['Open', 'High', 'Low', 'Close']
    data = data.drop(columns=['Volume'], errors='ignore')

    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Missing one or more required columns: {required_columns}")
    
    data = data[data['Close'] != 0]
    
    # Convert timezone to a standard format if present
    if data.index.tzinfo is not None:
        data.index = data.index.tz_localize(None)

    data['hour'] = data.index.hour
    data['1h_open'] = data.groupby(data.index.to_period('H'))['Open'].transform('first')
    return data

def buy_signal(data):
    buy_signals = []
    for i in range(1, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i-1]
        if row['hour'] != prev_row['hour']:
            current_hour_open = row['1h_open']
            lower_zone = current_hour_open * (1 - zone_width)
            if row['Close'] < lower_zone:
                buy_signals.append(row.name)
    return buy_signals

def sell_signal(data, buy_price, i):
    stop_loss_price = buy_price * (1 - stop_loss / 100)
    target_price = buy_price * (1 + target_profit / 100)
    current_price = data.iloc[i]['Close']  # Corrected to access 'Close' price

    sell_signals = []
    for j in range(1, len(data)):
        row = data.iloc[j]
        prev_row = data.iloc[j-1]
        if row['hour'] != prev_row['hour']:
            current_hour_open = row['1h_open']
            upper_zone = current_hour_open * (1 + zone_width)
            if row['Close'] > upper_zone:
                sell_signals.append(row.name)

    signals = sell_signals
    stop_loss_triggered = current_price <= stop_loss_price
    target_profit_reached = current_price >= target_price

    if target_profit_reached or stop_loss_triggered or signals:
        if target_profit_reached:
            return current_price, current_price - buy_price, 3  # Target profit reached
        elif stop_loss_triggered:
            return current_price, current_price - buy_price, 2  # Stop loss reached
        elif signals:
            return current_price, current_price - buy_price, 1  # EMA strategy signal

    return None, None, None

def trading_signal(data):
    buy_signals = buy_signal(data)
    data['Buy_Signal'] = np.where(data.index.isin(buy_signals), 1, 0)
    data['Sell_Signal'] = 0 
    return data

class HourlyStrategy(Strategy):
    stop_loss = stop_loss
    target_profit = target_profit

    def init(self):
        self.buy_price = None
        self.stop_loss_price = None
        self.target_price = None

    def next(self):
        if self.buy_price is None:
            if self.data.Buy_Signal[-1] == 1:
                self.buy_price = self.data.Close[-1]
                self.stop_loss_price = self.buy_price * (1 - stop_loss / 100)
                self.target_price = self.buy_price * (1 + target_profit / 100)
                self.buy()
        else:
            if (self.data.Sell_Signal[-1] == 1 or 
                self.data.Close[-1] <= self.stop_loss_price or 
                self.data.Close[-1] >= self.target_price):
                self.sell()
                self.buy_price = None
                self.stop_loss_price = None
                self.target_price = None

def process_stock(stock_name, csv_file_path):
    try:
        data = get_stock_data(csv_file_path)
        data = trading_signal(data)
        bt = Backtest(data, HourlyStrategy, cash=cash, commission=commission, exclusive_orders=exclusive_orders)
        output = bt.run()
        
        start_time_formatted = output['Start'].strftime('%Y-%m-%d %H:%M:%S')
        end_time_formatted = output['End'].strftime('%Y-%m-%d %H:%M:%S')
        metrics = {
            'Stock': stock_name,
            'Start': start_time_formatted,
            'End': end_time_formatted,
            'Duration': output['Duration'],
            'Exposure Time [%]': output['Exposure Time [%]'],
            'Equity Final [$]': output['Equity Final [$]'],
            'Equity Peak [$]': output['Equity Peak [$]'],
            'Return [%]': output['Return [%]'],
            'Buy & Hold Return [%]': output['Buy & Hold Return [%]'],
            'Return (Ann.) [%]': output['Return (Ann.) [%]'],
            'Volatility (Ann.) [%]': output['Volatility (Ann.) [%]'],
            'Sharpe Ratio': output['Sharpe Ratio'],
            'Sortino Ratio': output['Sortino Ratio'],
            'Calmar Ratio': output['Calmar Ratio'],
            'Max. Drawdown [%]': output['Max. Drawdown [%]'],
            'Avg. Drawdown [%]': output['Avg. Drawdown [%]'],
            'Max. Drawdown Duration': output['Max. Drawdown Duration'],
            'Avg. Drawdown Duration': output['Avg. Drawdown Duration'],
            '# Trades': output['# Trades'],
            'Win Rate [%]': output['Win Rate [%]'],
            'Best Trade [%]': output['Best Trade [%]'],
            'Worst Trade [%]': output['Worst Trade [%]'],
            'Avg. Trade [%]': output['Avg. Trade [%]'],
            'Max. Trade Duration': output['Max. Trade Duration'],
            'Avg. Trade Duration': output['Avg. Trade Duration'],
            'Profit Factor': output['Profit Factor'],
            'Expectancy [%]': output['Expectancy [%]'],
            'SQN': output['SQN']
        }
        rounded_metrics = {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
        return pd.DataFrame([rounded_metrics])
    except Exception as e:
        return pd.DataFrame([{'Stock': stock_name, 'Error': str(e)}])

def main():
    start_time = time.time()
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_stock, stock, path) for stock, path in csv_files.items()]
        results = [future.result() for future in futures]
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Total Time Taken: {total_duration:.2f} seconds")
    
    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv('summary.csv', index=False)
    print("Backtest results saved to 'summary.csv'")

if __name__ == '__main__':
    main()
