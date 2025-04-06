import numpy as np
import pandas as pd
import talib as ta_lib
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor
import time
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Define stock symbols and CSV file paths
csv_files = {
    'TATAMTR': '/home/ubuntu/Desktop/stock/stockmkt/data/1day/day/TATAMTRDVR.csv',
    'WIPRO': '/home/ubuntu/Desktop/stock/stockmkt/data/1day/day/WIPRO.csv',
    'VGUARD': '/home/ubuntu/Desktop/stock/stockmkt/data/1day/day/VGUARD.csv',
    'CGCL': '/home/ubuntu/Desktop/stock/stockmkt/data/1day/day/CGCL.csv',
    'BPL'   : '/home/ubuntu/Desktop/stock/stockmkt/data/1day/day/BPL.csv',
    'APOLLO': '/home/ubuntu/Desktop/stock/stockmkt/data/1day/day/APOLLO.csv',
}
stop_loss = 5
target_profit = 10
cash = 10000
commission = .002
exclusive_orders = True

def get_stock_data(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data = data.drop(columns=['volume'], errors='ignore')  
    return data

def ema_indicator(data, length=9, source='close', offset=0):
    src = data[source]
    ema = ta_lib.EMA(src, timeperiod=length)
    return ema.shift(offset)

def ma(source, length, ma_type, data):
    if ma_type == "SMA":
        return ta_lib.SMA(source, timeperiod=length)
    elif ma_type == "EMA":
        return ta_lib.EMA(source, timeperiod=length)
    elif ma_type == "SMMA (RMA)":
        return ta_lib.RMA(source, timeperiod=length)
    elif ma_type == "WMA":
        return ta_lib.WMA(source, timeperiod=length)
    elif ma_type == "VWMA":
        return (source * data['volume']).rolling(window=length).sum() / data['volume'].rolling(window=length).sum()

def calculate_indicators(data, ema_length=9, smoothing_type="SMA", smoothing_length=5, source='close', offset=0):
    ema = ema_indicator(data, length=ema_length, source=source, offset=offset)
    smoothing_line = ma(ema, smoothing_length, smoothing_type, data)
    return ema, smoothing_line

def buy_signal(df, i):
    if df.iloc[i]['EMA'] > df.iloc[i]['Smoothing_Line'] and df.iloc[i-1]['EMA'] <= df.iloc[i-1]['Smoothing_Line']:
        return df.iloc[i]['close']
    return None

def sell_signal(df, i, buy_price):
    current_price = df.iloc[i]['close']
    stop_loss_price = buy_price * (1 - stop_loss / 100)
    target_price = buy_price * (1 + target_profit / 100)

    ema_signal = df.iloc[i]['EMA'] < df.iloc[i]['Smoothing_Line'] and df.iloc[i-1]['EMA'] >= df.iloc[i-1]['Smoothing_Line']
    stop_loss_triggered = current_price <= stop_loss_price
    target_profit_reached = current_price >= target_price

    if target_profit_reached or stop_loss_triggered or ema_signal:
        if target_profit_reached:
            return current_price, current_price - buy_price, 3  # Target profit reached
        elif stop_loss_triggered:
            return current_price, current_price - buy_price, 2  # Stop loss reached
        elif ema_signal:
            return current_price, current_price - buy_price, 1  # EMA strategy signal

    return None, None, None

def ema_strategy(df):
    df['Signal'] = '-'
    df['Profit'] = np.nan
    df['Loss'] = np.nan
    df['Buy_Price'] = np.nan
    df['current_price'] = np.nan
    df['Sell_Reason'] = '-' # Add Sell_Reason column

    buy_price = None
    trades = []
    sell_reasons = {1: 0, 2: 0, 3: 0}  # Initialize counts for each reason
    total_sells = 0  # Initialize total sell count

    for i in range(1, len(df)):
        if buy_price is None:  # Looking for a buy signal
            buy_price = buy_signal(df, i)
            if buy_price is not None:
                df.at[df.index[i], 'Signal'] = 'Buy'
                df.at[df.index[i], 'Buy_Price'] = buy_price
        else:  # Looking for a sell signal
            current_price, profit_loss, reason = sell_signal(df, i, buy_price)
            if current_price is not None:
                df.at[df.index[i], 'Signal'] = 'Sell'
                df.at[df.index[i], 'current_price'] = current_price
                df.at[df.index[i], 'Sell_Reason'] = reason  # Record reason
                if profit_loss > 0:
                    df.at[df.index[i], 'Profit'] = profit_loss
                else:
                    df.at[df.index[i], 'Loss'] = -profit_loss
                trades.append((buy_price, current_price, profit_loss))
                sell_reasons[reason] += 1  # Increment count for the reason
                total_sells += 1  # Increment total sell count
                buy_price = None  # Reset buy price after a sell

    total_trades = len(trades)
    profitable_trades = sum(1 for _, _, profit in trades if profit > 0)
    lossable_trades = total_trades - profitable_trades
    win_ratio = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0

    return df, total_trades, profitable_trades, lossable_trades, win_ratio, sell_reasons, total_sells

class EMAStrategy(Strategy):
    def init(self):
        self.ema = self.I(ta_lib.EMA, self.data.Close, timeperiod=9)
        self.smoothing_line = self.I(ta_lib.SMA, self.ema, timeperiod=5)

    def next(self):
        if crossover(self.ema, self.smoothing_line):
            self.buy()
        elif crossover(self.smoothing_line, self.ema):
            self.sell()
            
def process_stock(stock_name, csv_file_path):
    try:
        data = get_stock_data(csv_file_path)
        result = [f"\nProcessing {stock_name} - {os.path.basename(csv_file_path)}"]

        startdate = data.index.min().strftime('%Y-%m-%d')
        end_date = data.index.max().strftime('%Y-%m-%d')
        result.append(f"{stock_name} Start Date: {startdate}")
        result.append(f"{stock_name} End Date: {end_date}")

        total_candles = len(data)
        
        ema, smoothing_line = calculate_indicators(data, ema_length=5, smoothing_type="SMA", smoothing_length=20)
        data['EMA'] = ema
        data['Smoothing_Line'] = smoothing_line

        data, total_trades, profitable_trades, lossable_trades, win_ratio, sell_reasons, total_sells = ema_strategy(data)

        result.extend([
            f"{stock_name} Total Candles Read: {total_candles}",
            f"{stock_name} Total Trades: {total_trades}",
            f"{stock_name} Profitable Trades: {profitable_trades}",
            f"{stock_name} Lossable Trades: {lossable_trades}",
            f"{stock_name} Win Ratio: {win_ratio:.2f}%",
            f"{stock_name} Total Sells: {total_sells}"
        ])
        
        data['date'] = data.index.strftime('%Y-%m-%d')
        data['time'] = data.index.strftime('%H:%M:%S')  # Extract time from index

        result.append(f"\n{stock_name} DataFrame with Signals:")
        result.append(data[['date', 'time', 'close', 'Signal', 'Sell_Reason']].to_string(index=False))

        result.append(f"\n{stock_name} Sell Reason Counts:")
        result.append(f"Target Profit Reached: {sell_reasons[3]}")
        result.append(f"Stop Loss Reached: {sell_reasons[2]}")
        result.append(f"EMA Strategy Signal: {sell_reasons[1]}")

        data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        
        bt = Backtest(data, EMAStrategy, cash, commission, exclusive_orders)
        output = bt.run()

        metrics = {
            'Stock': stock_name,
            'Start': output['Start'],
            'End': output['End'],
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
            '# Trades': output['# Trades'],
            'Win Rate [%]': output['Win Rate [%]'],
            'Best Trade [%]': output['Best Trade [%]'],
            'Worst Trade [%]': output['Worst Trade [%]'],
            'Avg. Trade [%]': output['Avg. Trade [%]'],
            'Max. Trade Duration': output['Max. Trade Duration'],
            'Avg. Trade Duration': output['Avg. Trade Duration'],
            'Profit Factor': output['Profit Factor'],
            'Expectancy [%]': output['Expectancy [%]'],
            'SQN': output['SQN'],
            '_strategy': output['_strategy'],
            '_equity_curve': output['_equity_curve'],
            '_trades': output['_trades'],
        }
        
        result.append(f"\n{stock_name} Backtesting Metrics:")
        for key, value in metrics.items():
            result.append(f"{key}: {value}")

        return "\n".join(result)
    except FileNotFoundError as e:
        return str(e)

def main():
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda stock: process_stock(stock, csv_files[stock]), csv_files))
    end_time = time.time()

    for result in results:
        print(result)

    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
