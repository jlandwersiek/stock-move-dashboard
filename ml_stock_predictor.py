import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor

from ta.trend import MACD, ADXIndicator, CCIIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.volatility import BollingerBands, AverageTrueRange

def get_data_from_tradier(symbol, api_key):
    import requests
    url = f"https://api.tradier.com/v1/markets/history"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    params = {
        "symbol": symbol,
        "interval": "daily"
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    if not data or "history" not in data or not data["history"] or "day" not in data["history"]:
        raise ValueError("No historical data returned from Tradier API.")
    df = pd.DataFrame(data["history"]["day"])
    df["timestamp"] = pd.to_datetime(df["date"])
    df.set_index("timestamp", inplace=True)
    return df

def get_stock_data(symbol, api_choice, api_key):
    if api_choice == "Tradier":
        return get_data_from_tradier(symbol, api_key)
    elif api_choice == "YCharts":
        return get_data_from_ycharts(symbol, api_key)
    else:
        raise ValueError("Invalid API choice. Please select either Tradier or YCharts.")

def add_technical_indicators(df, window_size):
    df = df.copy()
    df['return'] = df['close'].pct_change()

    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['macd'] = MACD(df['close']).macd()
    df['macd_signal'] = MACD(df['close']).macd_signal()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
    df['williams'] = WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
    df['stoch'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    df['ema_10'] = EMAIndicator(df['close'], window=10).ema_indicator()
    df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['bb_bbm'] = BollingerBands(df['close']).bollinger_mavg()
    df['bb_bbh'] = BollingerBands(df['close']).bollinger_hband()
    df['bb_bbl'] = BollingerBands(df['close']).bollinger_lband()
    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()

    df['sma_custom'] = df['close'].rolling(window=window_size).mean()
    df['roc_custom'] = df['close'].pct_change(periods=window_size)

    for lag in range(1, 6):
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_lag_{lag}'] = df['return'].shift(lag)

    df['rolling_std_5'] = df['close'].rolling(window=5).std()
    df['rolling_std_10'] = df['close'].rolling(window=10).std()

    df = df.dropna()
    return df

def train_and_predict(symbol, window_size, horizon, api_choice, api_key):
    df = get_stock_data(symbol, api_choice, api_key)
    df = add_technical_indicators(df, window_size)

    price_change = df['close'].shift(-horizon) - df['close']
    threshold = 0.005
    df['target_class'] = np.where(price_change > threshold, 1, 0)
    df['target_pct_return'] = (df['close'].shift(-horizon) - df['close']) / df['close']

    df['target_pct_return'] = df['target_pct_return'].clip(lower=-0.25, upper=0.25)
    df['target_volatility'] = df['target_pct_return'].abs()
    df['magnitude_class'] = pd.qcut(df['target_volatility'], q=3, labels=[0, 1, 2])
    df.dropna(inplace=True)

    drop_cols = ['target_class', 'target_pct_return', 'target_volatility', 'magnitude_class', 'symbol', 'date', 'timestamp']
    features = df.drop(columns=[col for col in drop_cols if col in df.columns])
    X = features.select_dtypes(include=[np.number])
    X = X.fillna(0).astype(float)

    y_class = df['target_class']
    y_magnitude = df['magnitude_class']
    y_reg = df['target_pct_return']

    selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
    X_selected = selector.fit_transform(X, y_class)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X = pd.DataFrame(X_scaled)

    X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, stratify=y_class, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, shuffle=False)

    voting_model = VotingClassifier(estimators=[
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("xgb", XGBClassifier(eval_metric="logloss")),
        ("lgbm", LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42))
    ], voting='soft')

    voting_model.fit(X_train, y_class_train)
    y_class_probs = voting_model.predict_proba(X_test)[:, 1]
    y_class_pred = (y_class_probs >= 0.5).astype(int)

    reg_median = LGBMRegressor(objective='quantile', alpha=0.5)
    reg_lower = LGBMRegressor(objective='quantile', alpha=0.1)
    reg_upper = LGBMRegressor(objective='quantile', alpha=0.9)

    reg_median.fit(X_reg_train, y_reg_train)
    reg_lower.fit(X_reg_train, y_reg_train)
    reg_upper.fit(X_reg_train, y_reg_train)

    y_reg_pred = reg_median.predict(X_reg_test)
    pred_lower = reg_lower.predict(X_reg_test)
    pred_upper = reg_upper.predict(X_reg_test)

    start_idx = len(df) - len(y_reg_test)
    close_prices_pred = df['close'].iloc[start_idx:].values
    close_prices_test = df['close'].iloc[start_idx:].values

    y_reg_pred_pct = (np.exp(y_reg_pred) - 1)
    pred_lower_pct = (np.exp(pred_lower) - 1)
    pred_upper_pct = (np.exp(pred_upper) - 1)
    y_reg_pred_usd = y_reg_pred_pct * close_prices_pred

    conflict_flags = []
    for cls_prob, cls_pred, reg_usd in zip(y_class_probs[:len(y_reg_pred_usd)], y_class_pred[:len(y_reg_pred_usd)], y_reg_pred_usd):
        if (cls_pred == 1 and cls_prob > 0.7 and reg_usd < -1.0) or (cls_pred == 0 and cls_prob > 0.7 and reg_usd > 1.0):
            conflict_flags.append(True)
        else:
            conflict_flags.append(False)

    mag_len = min(len(y_class_test), len(y_magnitude))
    return voting_model, reg_median, df, X_test, y_class_test.reset_index(drop=True), y_class_pred, y_reg_test.reset_index(drop=True), y_reg_pred_pct, [
        ("VotingClassifier", accuracy_score(y_class_test, y_class_pred))
    ], conflict_flags, close_prices_pred, close_prices_test, y_class_probs, pred_lower_pct, pred_upper_pct, y_magnitude[-mag_len:].values
