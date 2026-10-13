from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd


WEIGHTS = {
    "news_mood": 0.30,
    "volatility_mood": 0.25,
    "internals_mood": 0.25,
    "momentum_mood": 0.20,
}
REGIME_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def _normalize_columns(frame: pd.DataFrame, deduplicate_dates: bool = True) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().replace(" ", "_") for column in result.columns]
    aliases = {
        "date": "Date", "datetime": "Date", "timestamp": "Date",
        "published_at": "Date", "published_date": "Date", "news_date": "Date",
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "india_vix": "IndiaVIX", "vix": "IndiaVIX",
        "advance_decline_ratio": "advance_decline_ratio",
        "above_50dma_pct": "above_50dma_pct",
        "delivery_pct": "delivery_pct", "delivery_percentage": "delivery_pct",
    }
    rename = {}
    for column in result.columns:
        key = str(column).strip().lower().replace(" ", "_")
        if key in aliases:
            rename[column] = aliases[key]
    result = result.rename(columns=rename)
    if "Date" not in result:
        raise ValueError("Each input file needs a Date column (or date/datetime).")
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    result = result.dropna(subset=["Date"]).sort_values("Date")
    if deduplicate_dates:
        result = result.drop_duplicates("Date", keep="last")
    return result


def _pick(frame: pd.DataFrame | None, candidates: tuple[str, ...]) -> str | None:
    if frame is None:
        return None
    lookup = {str(column).lower(): column for column in frame.columns}
    return next((lookup[name.lower()] for name in candidates if name.lower() in lookup), None)


def _numeric_sentiment(news: pd.DataFrame) -> pd.Series:
    score_column = _pick(news, ("sentiment_score", "score", "polarity"))
    if score_column:
        scores = pd.to_numeric(news[score_column], errors="coerce")
        if scores.dropna().between(0, 2).all() and scores.dropna().min() >= 0:
            scores = scores - 1
        return scores.clip(-1, 1)

    label_column = _pick(news, ("sentiment", "label", "sentiment_label"))
    if not label_column:
        raise ValueError("News needs a sentiment/label column or numeric sentiment_score. Install the FinBERT extra to score headlines.")
    values = news[label_column]
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return (numeric - 1).clip(-1, 1)
    labels = values.astype(str).str.strip().str.lower()
    mapping = {
        "positive": 1.0, "bullish": 1.0, "positive sentiment": 1.0,
        "neutral": 0.0, "neutral sentiment": 0.0,
        "negative": -1.0, "bearish": -1.0, "negative sentiment": -1.0,
    }
    return labels.map(mapping)


def load_historical_headlines() -> pd.DataFrame:
    """Load the published 42K HF CSV; this source has no row-level dates."""
    url = (
        "https://huggingface.co/datasets/pranali96/indian_financial_news_42k/"
        "resolve/main/training_data_merged.csv"
    )
    news = pd.read_csv(url, low_memory=False)
    required = {"headline", "sentiment_label"}
    missing = required - set(news.columns)
    if missing:
        raise ValueError(f"Historical headline dataset is missing columns: {sorted(missing)}")
    news = news.rename(columns={"sentiment_label": "sentiment"})
    news["_sentiment_score"] = _numeric_sentiment(news)
    return news


def sentiment_label_counts(news: pd.DataFrame) -> pd.DataFrame:
    """Count positive, neutral, and negative labels without date attribution."""
    scores = _numeric_sentiment(news).dropna()
    labels = pd.Series(np.select(
        [scores > 0, scores < 0], ["Positive", "Negative"], default="Neutral"
    ), index=news.index, name="Sentiment")
    counts = labels.value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    return counts.rename_axis("Sentiment").reset_index(name="Headlines")


def aggregate_daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """Count daily positive/neutral/negative labels from dated headline rows."""
    date_columns = {
        str(column).strip().lower().replace(" ", "_") for column in news.columns
    }
    if not date_columns.intersection({"date", "datetime", "timestamp", "published_at", "published_date", "news_date"}):
        raise ValueError("Daily aggregation needs a row-level publication date; this dataset has none.")
    dated = _normalize_columns(news, deduplicate_dates=False)
    dated["_score"] = _numeric_sentiment(dated)
    dated = dated.dropna(subset=["_score"])
    dated["positive_count"] = (dated["_score"] > 0).astype(int)
    dated["neutral_count"] = (dated["_score"] == 0).astype(int)
    dated["negative_count"] = (dated["_score"] < 0).astype(int)
    daily = dated.groupby("Date", as_index=False).agg(
        positive_count=("positive_count", "sum"),
        neutral_count=("neutral_count", "sum"),
        negative_count=("negative_count", "sum"),
        total_headlines=("_score", "size"),
        daily_sentiment_score=("_score", "mean"),
    )
    return daily


def score_headlines_finbert(news: pd.DataFrame) -> pd.DataFrame:
    """Score headlines with ProsusAI/finbert; requires the optional FinBERT extra."""
    headline_column = _pick(news, ("headline", "title", "text", "news"))
    if not headline_column:
        raise ValueError("News file needs a headline, title, or text column.")
    try:
        pipeline = importlib.import_module("transformers").pipeline
    except ImportError as exc:
        raise RuntimeError("FinBERT is optional. Install it with: pip install -r requirements-finbert.txt") from exc

    scorer = pipeline("text-classification", model="ProsusAI/finbert", truncation=True)
    result = news.copy()
    predictions = scorer(result[headline_column].fillna("").astype(str).tolist(), batch_size=16)
    result["sentiment"] = [item["label"].lower() for item in predictions]
    result["sentiment_confidence"] = [float(item["score"]) for item in predictions]
    return result


def _rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    def percentile_last(values: np.ndarray) -> float:
        return float(np.mean(values <= values[-1]))

    return series.rolling(window, min_periods=30).apply(percentile_last, raw=True)


def classify_regime(score: float | pd.Series) -> str | pd.Series:
    bins = [-np.inf, 20, 40, 60, 80, np.inf]
    if isinstance(score, pd.Series):
        return pd.cut(score, bins=bins, labels=REGIME_ORDER, right=False).astype("object")
    value = float(score)
    if value < 20:
        return "Extreme Fear"
    if value < 40:
        return "Fear"
    if value < 60:
        return "Neutral"
    if value < 80:
        return "Greed"
    return "Extreme Greed"


def build_mood_frame(
    market: pd.DataFrame,
    vix: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
    internals: pd.DataFrame | None = None,
    mmi: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Align daily inputs and compute causal mood features on market sessions."""
    prices = _normalize_columns(market)
    if "Close" not in prices:
        raise ValueError("Nifty data needs a Close column.")
    for column in ("Open", "High", "Low", "Volume"):
        if column not in prices:
            prices[column] = np.nan

    frame = prices[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    if vix is not None and not vix.empty:
        vix_frame = _normalize_columns(vix)
        vix_col = _pick(vix_frame, ("IndiaVIX", "Close", "vix_close"))
        if vix_col:
            frame = frame.merge(vix_frame[["Date", vix_col]].rename(columns={vix_col: "IndiaVIX"}), on="Date", how="left")

    if news is not None and not news.empty:
        news_frame = _normalize_columns(news, deduplicate_dates=False)
        news_frame["_score"] = _numeric_sentiment(news_frame)
        news_frame = news_frame.dropna(subset=["_score"])
        news_frame["_positive"] = (news_frame["_score"] > 0).astype(int)
        news_frame["_negative"] = (news_frame["_score"] < 0).astype(int)
        news_daily = news_frame.groupby("Date").agg(
            daily_sentiment_score=("_score", "mean"),
            news_count=("_score", "size"),
            positive_count=("_positive", "sum"),
            negative_count=("_negative", "sum"),
        ).reset_index()
        frame = frame.merge(news_daily, on="Date", how="left")
        frame["news_count"] = frame["news_count"].fillna(0)
        frame["positive_count"] = frame["positive_count"].fillna(0)
        frame["negative_count"] = frame["negative_count"].fillna(0)
        sentiment = frame["daily_sentiment_score"]
        frame["sentiment_momentum"] = sentiment - sentiment.rolling(5, min_periods=2).mean()
        frame["sentiment_volatility"] = sentiment.rolling(5, min_periods=3).std()
        frame["news_mood"] = ((sentiment + 1) * 50).clip(0, 100)

    if "IndiaVIX" in frame:
        frame["IndiaVIX"] = pd.to_numeric(frame["IndiaVIX"], errors="coerce")
        frame["vix_change"] = frame["IndiaVIX"].pct_change()
        frame["volatility_mood"] = (1 - _rolling_percentile(frame["IndiaVIX"])) * 100

    if internals is not None and not internals.empty:
        internal_frame = _normalize_columns(internals)
        internal_frame = internal_frame.drop(columns=[column for column in ("Open", "High", "Low", "Close", "Volume") if column in internal_frame])
        frame = frame.merge(internal_frame, on="Date", how="left", suffixes=("", "_internal"))

    internal_components = []
    if "advance_decline_ratio" in frame:
        ratio = pd.to_numeric(frame["advance_decline_ratio"], errors="coerce").clip(lower=0)
        frame["ad_mood"] = 100 * ratio / (1 + ratio)
        internal_components.append("ad_mood")
    if "above_50dma_pct" in frame:
        above = pd.to_numeric(frame["above_50dma_pct"], errors="coerce")
        if above.dropna().max() <= 1:
            above = above * 100
        frame["breadth_mood"] = above.clip(0, 100)
        internal_components.append("breadth_mood")
    if "delivery_pct" in frame:
        delivery = pd.to_numeric(frame["delivery_pct"], errors="coerce")
        frame["delivery_mood"] = _rolling_percentile(delivery) * 100
        internal_components.append("delivery_mood")
    if internal_components:
        frame["internals_mood"] = frame[internal_components].mean(axis=1)

    frame["return_1"] = frame["Close"].pct_change()
    frame["return_5"] = frame["Close"].pct_change(5)
    frame["return_20"] = frame["Close"].pct_change(20)
    frame["momentum_mood"] = _rolling_percentile(frame["return_20"]) * 100
    delta = frame["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    frame["rsi"] = (100 - 100 / (1 + relative_strength)).fillna(50)
    fast = frame["Close"].ewm(span=12, adjust=False).mean()
    slow = frame["Close"].ewm(span=26, adjust=False).mean()
    frame["macd_hist"] = (fast - slow) - (fast - slow).ewm(span=9, adjust=False).mean()

    available = [name for name in WEIGHTS if name in frame]
    weighted = sum(frame[name] * WEIGHTS[name] for name in available)
    row_weight = sum(frame[name].notna() * WEIGHTS[name] for name in available)
    frame["mood_score"] = (weighted / row_weight.replace(0, np.nan)).clip(0, 100)
    frame["mood_momentum"] = frame["mood_score"] - frame["mood_score"].rolling(5, min_periods=2).mean()
    frame["mood_regime"] = classify_regime(frame["mood_score"])
    frame["next_day_return"] = frame["Close"].pct_change().shift(-1)
    frame["target_up"] = (frame["next_day_return"] > 0).astype("float")
    frame.loc[frame["next_day_return"].isna(), "target_up"] = np.nan

    if mmi is not None and not mmi.empty:
        mmi_frame = _normalize_columns(mmi)
        mmi_col = _pick(mmi_frame, ("MMI", "MarketMoodIndex", "MoodIndex", "Value", "Close"))
        if mmi_col:
            raw = pd.to_numeric(mmi_frame[mmi_col], errors="coerce")
            finite = raw.dropna()
            if not finite.empty and finite.min() < 0:
                raw = (raw - finite.min()) / (finite.max() - finite.min()) * 100
            frame = frame.merge(
                pd.DataFrame({"Date": mmi_frame["Date"], "mmi_score": raw}),
                on="Date", how="left",
            )

    return frame.replace([np.inf, -np.inf], np.nan).sort_values("Date").reset_index(drop=True)


def make_demo_inputs(seed: int = 24, days: int = 900) -> dict[str, pd.DataFrame]:
    """Generate deterministic, explicitly synthetic data for a UI/demo walkthrough."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-12-31", periods=days)
    cycles = np.sin(np.linspace(0, 22, days))
    daily_returns = 0.00035 + cycles * 0.0012 + rng.normal(0, 0.009, days)
    close = 17000 * np.exp(np.cumsum(daily_returns))
    open_price = close * (1 + rng.normal(0, 0.002, days))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.009, days))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.008, days))
    market = pd.DataFrame({"Date": dates, "Open": open_price, "High": high, "Low": low, "Close": close,
                           "Volume": rng.integers(300_000, 900_000, days)})
    vix_values = np.clip(17 - cycles * 4 + rng.normal(0, 2.2, days), 9, 48)
    vix = pd.DataFrame({"Date": dates, "Close": vix_values})
    sentiment_base = np.clip(cycles * 0.55 + rng.normal(0, 0.28, days), -1, 1)
    labels = np.where(sentiment_base > 0.18, "positive", np.where(sentiment_base < -0.18, "negative", "neutral"))
    news = pd.DataFrame({
        "Date": dates, "headline": [f"Synthetic demo headline {i + 1:04d}" for i in range(days)],
        "sentiment": labels,
    })
    internals = pd.DataFrame({
        "Date": dates,
        "advance_decline_ratio": np.clip(1 + cycles * 0.75 + rng.normal(0, 0.3, days), 0.05, None),
        "above_50dma_pct": np.clip(50 + cycles * 27 + rng.normal(0, 11, days), 3, 97),
        "delivery_pct": np.clip(35 + cycles * 8 + rng.normal(0, 5, days), 10, 70),
    })
    mmi = pd.DataFrame({"Date": dates, "MMI": np.clip(50 + sentiment_base * 31 + rng.normal(0, 8, days), 0, 100)})
    return {"market": market, "vix": vix, "news": news, "internals": internals, "mmi": mmi}