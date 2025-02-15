import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import talib as ta_lib
from datetime import date, datetime
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from concurrent.futures import ThreadPoolExecutor
import time


# Define stock symbols and CSV file paths
csv_files = {
    'WIPRO': '/home/ubuntu/Desktop/stock/stockmkt/data/3minute/WIPRO.csv',
    'ZOMATO': '/home/ubuntu/Desktop/stock/stockmkt/data/3minute/ZOMATO.csv'
}

def get_stock_data(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    return data

def calculate_ema(data, short_length, long_length):
    data['EMA_5'] = ta_lib.EMA(data['close'].values, timeperiod=short_length)
    data['EMA_20'] = ta_lib.EMA(data['close'].values, timeperiod=long_length)
    return data

def buy_signal(df, i):
    if df.iloc[i]['EMA_5'] > df.iloc[i]['EMA_20'] and df.iloc[i-1]['EMA_5'] <= df.iloc[i-1]['EMA_20']:
        df.at[df.index[i], 'Signal'] = 'Buy'
        return df.iloc[i]['close']
    return None

def sell_signal(df, i, buy_price):
    if df.iloc[i]['EMA_5'] < df.iloc[i]['EMA_20'] and df.iloc[i-1]['EMA_5'] >= df.iloc[i-1]['EMA_20']:
        df.at[df.index[i], 'Signal'] = 'Sell'
        sell_price = df.iloc[i]['close']
        profit_loss = sell_price - buy_price
        if profit_loss > 0:
            df.at[df.index[i], 'Profit'] = profit_loss
        else:
            df.at[df.index[i], 'Loss'] = -profit_loss
        return sell_price, profit_loss
    return None, None

def ema_strategy(df):
    df['Signal'] = ''
    df['Profit'] = np.nan
    df['Loss'] = np.nan

    buy_price = None
    trades = []

    for i in range(1, len(df)):
        if buy_price is None:  # Looking for a buy signal
            buy_price = buy_signal(df, i)
        else:  # Looking for a sell signal
            sell_price, profit_loss = sell_signal(df, i, buy_price)
            if sell_price is not None:
                trades.append((buy_price, sell_price, profit_loss))
                buy_price = None  # Reset buy price after a sell

    total_trades = len(trades)
    profitable_trades = sum(1 for _, _, profit in trades if profit > 0)
    lossable_trades = total_trades - profitable_trades
    win_ratio = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0

    return df, total_trades, profitable_trades, lossable_trades, win_ratio

class EMAStrategy(Strategy):
    short_length = 5
    long_length = 20

    def init(self):
        self.ema_5 = self.I(ta_lib.EMA, self.data.Close, self.short_length)
        self.ema_20 = self.I(ta_lib.EMA, self.data.Close, self.long_length)

    def next(self):
        if crossover(self.ema_5, self.ema_20):
            self.buy()
        elif crossover(self.ema_20, self.ema_5):
            self.sell()

def process_stock(stock_name, csv_file_path):
    try:
        data = get_stock_data(csv_file_path)
        result = []
        result.append(f"Processing {stock_name} - {os.path.basename(csv_file_path)}")
        startdate = data.index.min().strftime('%Y-%m-%d')
        end_date = data.index.max().strftime('%Y-%m-%d')
        result.append(f"{stock_name} Start Date: {startdate}")
        result.append(f"{stock_name} End Date: {end_date}")

        total_candles = len(data)
        
        data = calculate_ema(data, short_length=5, long_length=20)
        
        data, total_trades, profitable_trades, lossable_trades, win_ratio = ema_strategy(data)

        result.append(f"{stock_name} Total Candles Read: {total_candles}")
        result.append(f"{stock_name} Total Trades: {total_trades}")
        result.append(f"{stock_name} Profitable Trades: {profitable_trades}")
        result.append(f"{stock_name} Lossable Trades: {lossable_trades}")
        result.append(f"{stock_name} Win Ratio: {win_ratio:.2f}%")
        
        signals = data[data['Signal'] != '']
        if not signals.empty:
            result.append(f"\n{stock_name} Signals Detected:")
            result.append(signals[['close', 'EMA_5', 'EMA_20', 'Signal', 'Profit', 'Loss']].to_string())
        else:
            result.append(f"No buy or sell signals detected for {stock_name}.")
        
        # Prepare data for backtesting
        data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        
        bt = Backtest(data, EMAStrategy, cash=10000, commission=.002, exclusive_orders=True)
        output = bt.run()
        result.append(f"{stock_name} Backtest Result:\n{output}")

        return result
    except Exception as e:
        return [f"Error processing {stock_name}: {e}"]

def main():
    start_time = time.time()  # Start time
    results = []

    with ThreadPoolExecutor(max_workers=2) as executor:  # Adjust max_workers based on your system's capabilities
        futures = [executor.submit(process_stock, stock_name, csv_file_path) for stock_name, csv_file_path in csv_files.items()]
        
        for future in futures:
            results.append(future.result())  # Collect results

    for result in results:
        for line in result:
            print(line)

    end_time = time.time()  # End time
    print(f"Total Processing Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
