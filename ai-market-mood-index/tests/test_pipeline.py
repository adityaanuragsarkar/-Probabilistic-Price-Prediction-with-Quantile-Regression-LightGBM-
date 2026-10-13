import numpy as np
import pandas as pd
import pytest

from mood_index.data import (
    aggregate_daily_sentiment,
    build_mood_frame,
    classify_regime,
    make_demo_inputs,
    sentiment_label_counts,
)
from mood_index.model import backtest_rules, direction_signal


def test_demo_pipeline_produces_bounded_mood_and_next_day_target():
    inputs = make_demo_inputs(days=420)
    frame = build_mood_frame(**inputs)
    observed = frame["mood_score"].dropna()
    assert observed.between(0, 100).all()
    assert frame["target_up"].iloc[-1] != frame["target_up"].iloc[-1]
    assert frame["mood_regime"].dropna().isin(
        ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    ).all()


def test_optional_components_are_renormalized_not_fabricated():
    inputs = make_demo_inputs(days=360)
    frame = build_mood_frame(market=inputs["market"], news=inputs["news"])
    assert frame["mood_score"].notna().any()
    assert "internals_mood" not in frame
    assert "volatility_mood" not in frame


def test_regime_thresholds_and_probability_signal():
    assert classify_regime(19.9) == "Extreme Fear"
    assert classify_regime(20) == "Fear"
    assert classify_regime(80) == "Extreme Greed"
    assert direction_signal(0.56) == "BUY"
    assert direction_signal(0.45) == "HOLD"
    assert direction_signal(0.44) == "SELL"


def test_backtest_aligns_close_signal_to_next_day_return():
    inputs = make_demo_inputs(days=420)
    frame = build_mood_frame(**inputs)
    result = backtest_rules(frame)
    assert result["Date"].is_monotonic_increasing
    assert np.isfinite(result[["buy_hold", "momentum", "contrarian", "combined"]].to_numpy()).all()


def test_daily_sentiment_counts_group_numeric_labels_by_publication_date():
    headlines = pd.DataFrame({
        "published_at": ["2024-01-02 09:00", "2024-01-02 12:00", "2024-01-03 10:00"],
        "headline": ["bad", "flat", "good"],
        "sentiment_label": [0, 1, 2],
    })
    daily = aggregate_daily_sentiment(headlines).set_index("Date")
    first_day = daily.loc[pd.Timestamp("2024-01-02")]
    assert first_day["positive_count"] == 0
    assert first_day["neutral_count"] == 1
    assert first_day["negative_count"] == 1
    assert first_day["total_headlines"] == 2
    assert daily.loc[pd.Timestamp("2024-01-03"), "positive_count"] == 1


def test_historical_dataset_label_summary_does_not_require_dates():
    news = pd.DataFrame({"sentiment_label": [0, 1, 1, 2, 2, 2]})
    counts = sentiment_label_counts(news).set_index("Sentiment")["Headlines"]
    assert counts.to_dict() == {"Positive": 3, "Neutral": 2, "Negative": 1}


def test_daily_aggregation_rejects_undated_headlines():
    news = pd.DataFrame({"headline": ["A headline"], "sentiment_label": [2]})
    with pytest.raises(ValueError, match="row-level publication date"):
        aggregate_daily_sentiment(news)