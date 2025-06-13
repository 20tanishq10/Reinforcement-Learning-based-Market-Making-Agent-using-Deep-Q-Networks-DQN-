# data_preprocessing.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

LOB_DEPTH = 10  # configurable


def load_orderbook_data(orderbook_path):
    df = pd.read_csv(orderbook_path)
    df['minute'] = pd.to_datetime(df['minute'], format="%d-%m-%Y %H:%M")
    df['date'] = df['minute'].dt.strftime('%Y%m%d')
    return df

def load_trade_data(trade_path):
    df = pd.read_csv(trade_path)
    df['minute'] = pd.to_datetime(df['minute'], format="%d-%m-%Y %H:%M")
    df['date'] = df['minute'].dt.strftime('%Y%m%d')
    return df


def extract_lob_features(df):
    features = []
    for lvl in range(LOB_DEPTH):
        for prefix in ['bids_distance_', 'bids_market_notional_',
                       'asks_distance_', 'asks_market_notional_']:
            col = f"{prefix}{lvl}"
            if col in df.columns:
                features.append(col)
    return df[features]


def extract_market_features(df):
    return df[['midpoint', 'spread', 'buys', 'sells']]


def create_sequences(orderbook_df, trade_df, time_window=50):
    all_days = orderbook_df['date'].unique()
    data_by_day = {}
    scaler = MinMaxScaler()

    for day in all_days:
        ob_day = orderbook_df[orderbook_df['date'] == day].copy()
        trade_day = trade_df[trade_df['date'] == day].copy()

        ob_day = ob_day.sort_values('minute')
        trade_day = trade_day.sort_values('minute')

        ob_day.reset_index(drop=True, inplace=True)
        trade_day.reset_index(drop=True, inplace=True)

        lob_features = extract_lob_features(ob_day).values
        market_features = extract_market_features(ob_day).values

        X_lob, X_market = [], []
        for i in range(time_window, len(lob_features)):
            lob_seq = lob_features[i - time_window:i]
            X_lob.append(lob_seq)
            X_market.append(market_features[i])

        X_lob = np.array(X_lob)
        X_market = np.array(X_market)
        
        # Scale market features
        X_market = scaler.fit_transform(X_market)

        data_by_day[day] = {
            'lob': X_lob,
            'market': X_market,
        }
    return data_by_day


if __name__ == '__main__':
    ORDERBOOK_PATH = './data/orderbook.csv'
    TRADE_PATH = './data/trade.csv'
    
    ob_df = load_orderbook_data(ORDERBOOK_PATH)
    trade_df = load_trade_data(TRADE_PATH)

    dataset = create_sequences(ob_df, trade_df, time_window=50)

    # Optionally save for faster access later
    for day, data in dataset.items():
        np.savez_compressed(f'./processed/{day}.npz', lob=data['lob'], market=data['market'])

    print("Saved all processed days in './processed/' folder.")
