import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import date, datetime

# Define stock symbol and time intervals
stocksymbols = ['RELIANCE.NS']
startdate = date(2024, 7, 13)  # Adjust to the last 7 days from today
end_date = date(2024, 7, 20)
interval = '1m'

def get_stock_data(symbols, start, end, interval):
    data = yf.download(symbols, start=start, end=end, interval=interval)
    return data

def calculate_ema(data, short_length, long_length):
    data['EMA_5'] = ta.ema(data['Close'], length=short_length)
    data['EMA_20'] = ta.ema(data['Close'], length=long_length)
    return data

def buy_signal(df, i):
    if df.iloc[i]['EMA_5'] > df.iloc[i]['EMA_20'] and df.iloc[i-1]['EMA_5'] <= df.iloc[i-1]['EMA_20']:
        df.at[df.index[i], 'Signal'] = 'Buy'
        return df.iloc[i]['Close']
    return None

def sell_signal(df, i, buy_price):
    if df.iloc[i]['EMA_5'] < df.iloc[i]['EMA_20'] and df.iloc[i-1]['EMA_5'] >= df.iloc[i-1]['EMA_20']:
        df.at[df.index[i], 'Signal'] = 'Sell'
        sell_price = df.iloc[i]['Close']
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

def main():
    # Print start and end dates
    print(f"Start Date: {startdate.strftime('%Y-%m-%d')}")
    print(f"End Date: {end_date.strftime('%Y-%m-%d')}")

    # Get stock data
    data = get_stock_data(stocksymbols, startdate, end_date, interval)
    total_candles = len(data) 
    
    # Calculate EMA
    data = calculate_ema(data, short_length=5, long_length=20)
    
    # Apply EMA strategy
    data, total_trades, profitable_trades, lossable_trades, win_ratio = ema_strategy(data)

    # Display results
    print(f"Total Candles Read: {total_candles}")
    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {profitable_trades}")
    print(f"Lossable Trades: {lossable_trades}")
    print(f"Win Ratio: {win_ratio:.2f}%")

    # Show only signals detected
    signals = data[data['Signal'] != '']
    if not signals.empty:
        print("\nSignals Detected:")
        print(signals[['Close', 'EMA_5', 'EMA_20', 'Signal', 'Profit', 'Loss']])
    else:
        print("No buy or sell signals detected.")


if __name__ == "__main__":
    main()
