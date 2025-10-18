"""
CNN-LSTM vs Attention-LSTM baseline for crypto price & direction prediction
Fetches OHLCV-like data from CoinGecko, preprocesses, builds windows,
trains two models (CNN-LSTM and Attention-LSTM), evaluates regression (next-close)
and classification (direction up/down), and saves results & plots.

Usage:
    python3 cnn_lstm_vs_attention_lstm.py --coin bitcoin --vs_currency usd --days 365 --window 60 --epochs 30

Requirements:
    pip install requests pandas numpy scikit-learn tensorflow matplotlib

Notes:
- This script uses CoinGecko public /coins/{id}/market_chart endpoint.
- For futures-specific features (funding rate, open interest) you'll need exchange API.

Simple architecture diagrams (ASCII):

CNN-LSTM:
Input (window x features)
  -> Conv1D(filters=64, kernel=3)
  -> MaxPooling1D(pool=2)
  -> LSTM(64)
  -> Dense(32) -> Output

Attention-LSTM:
Input (window x features)
  -> LSTM(return_sequences=True)
  -> Attention Layer (weights over time)
  -> Dense(32) -> Output

"""

import argparse
import os
import json
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, LSTM, Dense, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# -------------------------
# Utilities: fetch & prepare
# -------------------------

def fetch_coingecko_market_chart(coin_id='bitcoin', vs_currency='usd', days=365, interval='daily'):
    """Fetch prices and volumes from CoinGecko market_chart endpoint.
    Returns a DataFrame with columns: timestamp, price, volume, market_cap
    Note: CoinGecko returns timestamps in ms.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': vs_currency, 'days': days, 'interval': interval}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    prices = data.get('prices', [])  # [ [timestamp, price], ... ]
    volumes = data.get('total_volumes', [])
    market_caps = data.get('market_caps', [])

    # align lengths
    n = min(len(prices), len(volumes), len(market_caps))
    rows = []
    for i in range(n):
        ts = int(prices[i][0] // 1000)
        dt = datetime.utcfromtimestamp(ts)
        price = float(prices[i][1])
        vol = float(volumes[i][1])
        mcap = float(market_caps[i][1])
        rows.append({'timestamp': dt, 'close': price, 'volume': vol, 'market_cap': mcap})

    df = pd.DataFrame(rows)
    df.set_index('timestamp', inplace=True)
    return df


def engineer_features(df):
    """Create OHLC-like and technical features from 'close' and 'volume'.
    Because CoinGecko returns only price series for 'market_chart', we use returns
    and some rolling stats as features.
    """
    df = df.copy()
    df['log_close'] = np.log(df['close'])
    df['ret'] = df['log_close'].diff()
    df['ret'].fillna(0, inplace=True)

    # rolling features
    for w in [3, 7, 14]:
        df[f'ret_mean_{w}'] = df['ret'].rolling(w).mean().fillna(0)
        df[f'ret_std_{w}'] = df['ret'].rolling(w).std().fillna(0)
    df['vol_mean_7'] = df['volume'].rolling(7).mean().fillna(0)

    # direction label (next-step)
    df['target_close'] = df['close'].shift(-1)
    df['target_ret'] = np.log(df['target_close']) - df['log_close']
    df['target_dir'] = (df['target_ret'] > 0).astype(int)

    df.dropna(inplace=True)
    return df


def create_windows(df, feature_cols, window=60, target_col='target_ret'):
    X = []
    y_reg = []
    y_clf = []
    for i in range(len(df) - window):
        X.append(df[feature_cols].iloc[i:i+window].values)
        y_reg.append(df[target_col].iloc[i+window-1])  # predict next-step return (aligned)
        y_clf.append(df['target_dir'].iloc[i+window-1])
    X = np.array(X)
    y_reg = np.array(y_reg)
    y_clf = np.array(y_clf)
    return X, y_reg, y_clf


# -------------------------
# Models
# -------------------------

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1],), initializer='random_normal', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time, features)
        # score each time step by dot-product with W
        scores = tf.tensordot(inputs, self.W, axes=1)  # (batch, time)
        weights = tf.nn.softmax(scores, axis=1)  # (batch, time)
        expanded = tf.expand_dims(weights, axis=-1)  # (batch, time, 1)
        context = tf.reduce_sum(inputs * expanded, axis=1)  # (batch, features)
        return context


def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_attention_lstm(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    att = AttentionLayer()(lstm_out)
    d1 = Dense(32, activation='relu')(att)
    out = Dense(1, activation='linear')(d1)
    model = Model(inputs, out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# -------------------------
# Training and evaluation
# -------------------------

def train_and_eval(X, y_reg, y_clf, params):
    # split train / val / test using time-order
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train_reg, y_val_reg, y_test_reg = y_reg[:train_end], y_reg[train_end:val_end], y_reg[val_end:]
    y_train_clf, y_val_clf, y_test_clf = y_clf[:train_end], y_clf[train_end:val_end], y_clf[val_end:]

    # scale features per feature axis
    nsamples, nt, nf = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, nf)
    scaler.fit(X_train_flat)
    X_train_s = scaler.transform(X_train_flat).reshape(nsamples, nt, nf)

    def scale_dataset(Xset):
        s = Xset.reshape(-1, nf)
        s = scaler.transform(s)
        return s.reshape(Xset.shape)

    X_val_s = scale_dataset(X_val)
    X_test_s = scale_dataset(X_test)

    # scale target for regression? we predict returns directly (no scaling)

    # Models
    input_shape = (X_train_s.shape[1], X_train_s.shape[2])

    cnn_model = build_cnn_lstm(input_shape)
    att_model = build_attention_lstm(input_shape)

    # callbacks
    os.makedirs(params['out_dir'], exist_ok=True)
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ck1 = ModelCheckpoint(os.path.join(params['out_dir'], 'cnn_model.h5'), save_best_only=True)
    ck2 = ModelCheckpoint(os.path.join(params['out_dir'], 'att_model.h5'), save_best_only=True)

    print('Training CNN-LSTM...')
    history_cnn = cnn_model.fit(X_train_s, y_train_reg, validation_data=(X_val_s, y_val_reg),
                                epochs=params['epochs'], batch_size=params['batch_size'], callbacks=[es, ck1], verbose=2)

    print('Training Attention-LSTM...')
    history_att = att_model.fit(X_train_s, y_train_reg, validation_data=(X_val_s, y_val_reg),
                                epochs=params['epochs'], batch_size=params['batch_size'], callbacks=[es, ck2], verbose=2)

    # Evaluate regression
    def eval_reg(model, Xs, y_true):
        y_pred = model.predict(Xs).flatten()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        return y_pred, mae, rmse

    y_cnn_pred, cnn_mae, cnn_rmse = eval_reg(cnn_model, X_test_s, y_test_reg)
    y_att_pred, att_mae, att_rmse = eval_reg(att_model, X_test_s, y_test_reg)

    # Classification (direction): threshold predicted returns > 0
    cnn_dir_pred = (y_cnn_pred > 0).astype(int)
    att_dir_pred = (y_att_pred > 0).astype(int)
    cnn_acc = accuracy_score(y_test_clf, cnn_dir_pred)
    att_acc = accuracy_score(y_test_clf, att_dir_pred)

    results = {
        'cnn': {'mae': cnn_mae, 'rmse': cnn_rmse, 'acc': cnn_acc, 'y_pred': y_cnn_pred.tolist()},
        'att': {'mae': att_mae, 'rmse': att_rmse, 'acc': att_acc, 'y_pred': y_att_pred.tolist()}
    }

    # save results
    with open(os.path.join(params['out_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # plots
    idx = np.arange(len(y_test_reg))
    plt.figure(figsize=(12,6))
    plt.plot(idx, y_test_reg, label='true_return')
    plt.plot(idx, y_cnn_pred, label='cnn_pred', alpha=0.8)
    plt.plot(idx, y_att_pred, label='att_pred', alpha=0.8)
    plt.legend()
    plt.title('Test set: true vs predicted returns')
    plt.savefig(os.path.join(params['out_dir'], 'pred_returns.png'))
    plt.close()

    # save scaler
    import joblib
    joblib.dump(scaler, os.path.join(params['out_dir'], 'scaler.pkl'))

    return results


# -------------------------
# Main
# -------------------------

def main(args):
    print('Fetching data from CoinGecko...')
    df = fetch_coingecko_market_chart(coin_id=args.coin, vs_currency=args.vs_currency, days=args.days, interval='daily')
    print(f'Fetched {len(df)} rows')

    df = engineer_features(df)
    print('Engineered features, sample:')
    print(df.head())

    feature_cols = ['close', 'volume', 'market_cap', 'ret', 'ret_mean_3', 'ret_std_3', 'ret_mean_7', 'ret_std_7', 'vol_mean_7']
    X, y_reg, y_clf = create_windows(df, feature_cols, window=args.window, target_col='target_ret')
    print('Created windows:', X.shape, y_reg.shape, y_clf.shape)

    params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'out_dir': args.out_dir
    }

    results = train_and_eval(X, y_reg, y_clf, params)
    print('Results saved to', args.out_dir)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coin', type=str, default='bitcoin')
    parser.add_argument('--vs_currency', type=str, default='usd')
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_dir', type=str, default='./output')
    args = parser.parse_args()
    main(args)
