from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = None
    for candidate in ["Date", "date", "Datetime", "timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError("Expected a date column named 'Date' or similar.")

    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).reset_index(drop=True)


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_bollinger_features(df: pd.DataFrame) -> pd.DataFrame:
    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std(ddof=0)
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    df["bb_middle_20"] = rolling_mean
    df["bb_upper_20"] = upper
    df["bb_lower_20"] = lower
    df["bb_position_20"] = (df["Close"] - lower) / (upper - lower).replace(0, np.nan)
    return df


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"atr_{window}"] = tr.rolling(window).mean()
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ret_1"] = df["Close"].pct_change().shift(1)
    df["ret_2"] = df["Close"].pct_change().shift(2)
    df["ret_3"] = df["Close"].pct_change().shift(3)
    df["ret_5"] = df["Close"].pct_change().shift(5)
    df["ret_10"] = df["Close"].pct_change().shift(10)

    df["vol_5"] = df["Close"].pct_change().rolling(5).std().shift(1)
    df["vol_10"] = df["Close"].pct_change().rolling(10).std().shift(1)
    df["vol_20"] = df["Close"].pct_change().rolling(20).std().shift(1)

    df["rsi_14"] = compute_rsi(df["Close"], window=14)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df = compute_bollinger_features(df)
    df = compute_atr(df, window=14)

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    if "Volume" in df.columns:
        df["vol_change_1"] = df["Volume"].pct_change().shift(1)
    else:
        df["vol_change_1"] = 0.0

    return df


def prepare_market_frame(nifty_df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    nifty = ensure_datetime(nifty_df)
    if "Close" not in nifty.columns:
        raise ValueError("Nifty DataFrame must include a Close column.")

    if vix_df is not None:
        vix = ensure_datetime(vix_df)
        vix_columns = ["Date", "Close"]
        missing = [c for c in vix_columns if c not in vix.columns]
        if missing:
            raise ValueError(f"VIX data missing required columns: {missing}")
        vix = vix[["Date", "Close"]].rename(columns={"Close": "IndiaVIX"})
        nifty = nifty.merge(vix, on="Date", how="left")

    nifty = compute_technical_indicators(nifty)

    if "IndiaVIX" in nifty.columns:
        nifty["india_vix_change_1"] = nifty["IndiaVIX"].pct_change().shift(1)
        nifty["india_vix_level"] = nifty["IndiaVIX"]
    else:
        nifty["india_vix_change_1"] = 0.0
        nifty["india_vix_level"] = 0.0

    nifty["next_close"] = nifty["Close"].shift(-1)
    return nifty


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = [
        col
        for col in df.columns
        if col not in {"Date", "Open", "High", "Low", "Close", "Volume", "next_close"}
        and df[col].dtype.kind in {"f", "i", "u", "b"}
    ]

    features = df[feature_cols].copy()
    target = df["next_close"].astype(float)

    features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    target = target.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    return features, target, feature_cols


def create_synthetic_market_data(n_rows: int = 2000, start_date: str = "2015-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start=start_date, periods=n_rows, freq="B")
    close = 17000 + np.cumsum(rng.normal(0, 120, n_rows))

    open_ = close * (1 + rng.normal(0, 0.002, n_rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, n_rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.010, n_rows))
    volume = rng.integers(500_000_000, 1_200_000_000, size=n_rows)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    return df


def create_synthetic_vix_data(n_rows: int = 2000, start_date: str = "2015-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range(start=start_date, periods=n_rows, freq="B")
    vix = 18 + np.cumsum(rng.normal(0, 2.5, n_rows))
    vix = np.clip(vix, 10, 80)
    return pd.DataFrame({"Date": dates, "Close": vix})
