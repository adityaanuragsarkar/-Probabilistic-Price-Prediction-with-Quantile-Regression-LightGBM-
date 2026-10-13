from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


BASE_FEATURES = [
    "mood_score", "mood_momentum", "IndiaVIX", "vix_change", "rsi",
    "macd_hist", "return_5", "sentiment_momentum", "sentiment_volatility",
    "news_count", "internals_mood", "momentum_mood",
]


def direction_features(frame: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in BASE_FEATURES if column in frame]
    features = frame[available].apply(pd.to_numeric, errors="coerce")
    regimes = pd.get_dummies(frame.get("mood_regime", pd.Series("Neutral", index=frame.index)), prefix="regime", dtype=float)
    for regime in ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"):
        column = "regime_" + regime
        if column not in regimes:
            regimes[column] = 0.0
    return pd.concat([features, regimes.reindex(columns=sorted(regimes.columns))], axis=1)


def train_direction_model(frame: pd.DataFrame) -> dict:
    """Train an 80/20 chronological holdout and refit on all labeled rows."""
    if "target_up" not in frame:
        raise ValueError("The frame needs target_up; build it from historical market data first.")
    X = direction_features(frame)
    y = pd.to_numeric(frame["target_up"], errors="coerce")
    valid = y.notna() & X.notna().all(axis=1)
    X, y = X.loc[valid], y.loc[valid].astype(int)
    if len(X) < 120:
        raise ValueError(f"Need at least 120 complete labeled sessions; found {len(X)}.")
    split = int(len(X) * 0.80)
    if y.iloc[:split].nunique() < 2 or y.iloc[split:].nunique() < 2:
        raise ValueError("Both Up and Down outcomes are needed in the chronological train and test periods.")

    settings = dict(
        objective="binary", n_estimators=350, learning_rate=0.035,
        num_leaves=15, max_depth=-1, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=2.0,
        random_state=42, verbosity=-1,
    )
    holdout_model = LGBMClassifier(**settings)
    holdout_model.fit(X.iloc[:split], y.iloc[:split])
    probabilities = holdout_model.predict_proba(X.iloc[split:])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y.iloc[split:], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y.iloc[split:], predictions)),
        "brier_score": float(brier_score_loss(y.iloc[split:], probabilities)),
        "roc_auc": float(roc_auc_score(y.iloc[split:], probabilities)),
        "test_sessions": int(len(probabilities)),
    }
    holdout = pd.DataFrame({"Date": frame.loc[X.index[split:], "Date"].to_numpy(),
                            "actual_up": y.iloc[split:].to_numpy(),
                            "p_up": probabilities}, index=X.index[split:])
    final_model = LGBMClassifier(**settings)
    final_model.fit(X, y)
    return {"model": final_model, "metrics": metrics, "holdout": holdout,
            "feature_columns": X.columns.tolist(), "samples": len(X)}


def predict_up_probability(model: LGBMClassifier, frame: pd.DataFrame) -> float:
    X = direction_features(frame)
    expected = list(model.feature_name_)
    X = X.reindex(columns=expected, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if X.empty or X.iloc[-1].isna().any():
        raise ValueError("The latest session does not have enough complete features for inference.")
    return float(model.predict_proba(X.tail(1))[:, 1][0])


def direction_signal(probability_up: float) -> str:
    if probability_up > 0.55:
        return "BUY"
    if probability_up < 0.45:
        return "SELL"
    return "HOLD"


def make_rule_signals(frame: pd.DataFrame) -> pd.DataFrame:
    mood = frame["mood_score"]
    rising = frame["mood_momentum"] > 0
    sentiment_turn = frame.get("sentiment_momentum", pd.Series(np.nan, index=frame.index)) > 0
    falling_sentiment = frame.get("sentiment_momentum", pd.Series(np.nan, index=frame.index)) < 0
    momentum = np.where((mood >= 60) & rising, 1, np.where((mood <= 40) & ~rising, -1, 0))
    contrarian = np.where((mood < 20) & sentiment_turn, 1,
                         np.where((mood >= 80) & falling_sentiment, -1, 0))
    combined = np.where(contrarian != 0, contrarian, momentum)
    return pd.DataFrame({"momentum": momentum, "contrarian": contrarian, "combined": combined}, index=frame.index)


def backtest_rules(frame: pd.DataFrame, fee_bps: float = 5.0) -> pd.DataFrame:
    """Close-to-close diagnostic backtest; signals at T earn the T+1 return."""
    signals = make_rule_signals(frame)
    next_return = pd.to_numeric(frame["next_day_return"], errors="coerce")
    result = pd.DataFrame({"Date": frame["Date"], "buy_hold": next_return}, index=frame.index)
    fee = fee_bps / 10_000
    for strategy in signals:
        position = signals[strategy].astype(float)
        turnover = position.diff().abs().fillna(position.abs())
        result[strategy] = position * next_return - turnover * fee
    return result.dropna(subset=["buy_hold"])


def equity_curves(backtest: pd.DataFrame) -> pd.DataFrame:
    curves = pd.DataFrame({"Date": backtest["Date"]})
    for strategy in ("buy_hold", "momentum", "contrarian", "combined"):
        if strategy in backtest:
            curves[strategy] = (1 + backtest[strategy].fillna(0)).cumprod()
    return curves