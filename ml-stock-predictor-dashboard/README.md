# ML Stock Predictor Dashboard

A machine learning model to forecast short-term price direction and return magnitude using Tradier API stock data and technical indicators.

## Features

- Pulls stock data from **Tradier API**
- Computes 15+ technical indicators (RSI, MACD, ATR, VWAP, etc.)
- Predicts:
  - **Direction** (Up/Down)
  - **Magnitude** (small, medium, large)
  - **Expected return range** using quantile regression
- Highlights **conflicting model signals**

## How to Use

```python
from ml_stock_predictor import train_and_predict

train_and_predict(
    symbol="SPY",
    window_size=5,
    horizon=1,
    api_choice="Tradier",
    api_key="YOUR_API_KEY"
)

## Requirements

pip install pandas numpy scikit-learn xgboost lightgbm ta requests
