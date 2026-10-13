from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mood_index.data import (
    REGIME_ORDER,
    WEIGHTS,
    aggregate_daily_sentiment,
    build_mood_frame,
    load_historical_headlines,
    make_demo_inputs,
    score_headlines_finbert,
    sentiment_label_counts,
)
from mood_index.model import (
    backtest_rules,
    direction_features,
    direction_signal,
    equity_curves,
    predict_up_probability,
    train_direction_model,
)


st.set_page_config(page_title="Market Mood Index", page_icon="◉", layout="wide")
IST = ZoneInfo("Asia/Kolkata")
COLORS = {
    "ink": "#1C332E", "muted": "#70807B", "paper": "#F3F6F2",
    "green": "#16795B", "mint": "#D9EFE4", "coral": "#E76F51",
    "gold": "#D5A93A", "line": "#D8E0DA", "blue": "#55788A",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {{ --ink:{COLORS['ink']}; --green:{COLORS['green']}; --paper:{COLORS['paper']}; }}
.stApp {{ background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }}
[data-testid="stSidebar"] {{ background:#E7ECE7; border-right:1px solid {COLORS['line']}; }}
.block-container {{ max-width:1380px; padding-top:1.6rem; padding-bottom:3rem; }}
h1,h2,h3 {{ font-family:'Space Grotesk',sans-serif !important; color:var(--ink); letter-spacing:0 !important; }}
.eyebrow {{ color:{COLORS['green']}; font:500 11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:1.2px; }}
.lede {{ color:{COLORS['muted']}; font-size:15px; max-width:780px; line-height:1.55; }}
.metric-label {{ color:{COLORS['muted']}; font:500 11px 'DM Mono',monospace; text-transform:uppercase; }}
.mood-value {{ font:600 52px 'Space Grotesk',sans-serif; line-height:1; color:var(--ink); }}
.mood-unit {{ font:500 15px 'DM Mono',monospace; color:{COLORS['muted']}; }}
.metric-rule {{ border-top:1px solid {COLORS['line']}; margin:10px 0 12px; }}
.regime-pill {{ display:inline-block; padding:5px 10px; font:500 11px 'DM Mono',monospace; border-radius:3px; background:{COLORS['mint']}; color:{COLORS['green']}; }}
.timestamp {{ color:{COLORS['muted']}; font:11px 'DM Mono',monospace; }}
.section-note {{ color:{COLORS['muted']}; font-size:12px; }}
.stButton>button {{ border-radius:3px; font-weight:600; }}
div[data-testid="stMetric"] {{ background:#fff; border:1px solid {COLORS['line']}; padding:14px 16px; border-radius:4px; }}
div[data-testid="stMetricLabel"] {{ font-family:'DM Mono',monospace; text-transform:uppercase; font-size:10px; }}
.footer-note {{ border-top:1px solid {COLORS['line']}; margin-top:2rem; padding-top:12px; color:{COLORS['muted']}; font-size:11px; }}
</style>
""", unsafe_allow_html=True)


def mood_gauge(score: float, regime: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "", "font": {"size": 36, "color": COLORS["ink"], "family": "Space Grotesk"}},
        title={"text": regime, "font": {"size": 14, "color": COLORS["muted"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickfont": {"color": COLORS["muted"], "size": 9}},
            "bar": {"color": COLORS["green"], "thickness": 0.24},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "#F4DCD5"},
                {"range": [20, 40], "color": "#F3E8D1"},
                {"range": [40, 60], "color": "#E8ECE7"},
                {"range": [60, 80], "color": "#DBEBE1"},
                {"range": [80, 100], "color": "#C6E4D4"},
            ],
            "threshold": {"line": {"color": COLORS["coral"], "width": 3}, "thickness": 0.8, "value": score},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=16, r=16, t=45, b=8), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def read_upload(upload) -> pd.DataFrame | None:
    if upload is None:
        return None
    return pd.read_csv(upload)


def get_inputs() -> tuple[dict[str, pd.DataFrame], bool]:
    mode = st.sidebar.radio("Data mode", ["Synthetic demo", "Upload historical CSVs"], index=0)
    if mode == "Synthetic demo":
        st.sidebar.caption("Generated data only. It is not actual market or news history.")
        return make_demo_inputs(), True

    with st.sidebar.expander("Add CSV sources", expanded=True):
        market_file = st.file_uploader("Nifty OHLCV (required)", type="csv", key="market")
        vix_file = st.file_uploader("India VIX", type="csv", key="vix")
        news_file = st.file_uploader("Daily news", type="csv", key="news")
        internals_file = st.file_uploader("Market internals", type="csv", key="internals")
        mmi_file = st.file_uploader("TickerTape MMI (validation only)", type="csv", key="mmi")
        if st.button("Load 42K historical headline labels", key="load_reference_news"):
            try:
                with st.spinner("Loading the public Hugging Face CSV..."):
                    st.session_state["reference_news"] = load_historical_headlines()
            except Exception as exc:
                st.error(f"Could not load the historical news dataset: {exc}")
    if market_file is None:
        st.info("Upload a Nifty CSV to build a historical mood index. The 42K news label summary can be viewed without it.")
    return {
        "market": read_upload(market_file), "vix": read_upload(vix_file),
        "news": read_upload(news_file), "internals": read_upload(internals_file),
        "mmi": read_upload(mmi_file),
    }, False


def show_data_quality(frame: pd.DataFrame, is_demo: bool) -> None:
    if is_demo:
        st.warning("Synthetic demo data is active. Scores, probabilities, and backtests below are illustrative only.", icon="⚠️")
    components = {
        "News sentiment": "news_mood", "India VIX": "volatility_mood",
        "Market internals": "internals_mood", "20-day momentum": "momentum_mood",
    }
    available = [label for label, column in components.items() if column in frame and frame[column].notna().any()]
    missing = [label for label in components if label not in available]
    if missing:
        st.info("Mood weights are redistributed across available components. Missing: " + ", ".join(missing) + ".")


def show_reference_headline_summary(news: pd.DataFrame) -> None:
    st.subheader("Historical headline labels")
    count_frame = sentiment_label_counts(news)
    counts = count_frame.set_index("Sentiment")["Headlines"]
    positive, neutral, negative = st.columns(3)
    positive.metric("Positive", f"{counts['Positive']:,}")
    neutral.metric("Neutral", f"{counts['Neutral']:,}")
    negative.metric("Negative", f"{counts['Negative']:,}")
    st.warning(
        "This published dataset has no per-headline publication date. These are "
        "overall label totals only; it cannot be assigned to trading days or used "
        "for the daily mood feature until dated headline timestamps are obtained.",
        icon="⚠️",
    )


def main() -> None:
    st.markdown('<div class="eyebrow">Daily research / India equities</div>', unsafe_allow_html=True)
    st.title("Market mood, made measurable.")
    st.markdown('<div class="lede">A daily composite of news tone, volatility, market breadth and price momentum, paired with a next-session direction estimate.</div>', unsafe_allow_html=True)

    inputs, is_demo = get_inputs()
    reference_news = st.session_state.get("reference_news")
    if inputs.get("market") is None:
        if reference_news is not None:
            show_reference_headline_summary(reference_news)
        st.info("Upload Nifty OHLCV data to align historical headlines with trading sessions.")
        st.stop()

    if st.sidebar.button("Score headlines with FinBERT", disabled=inputs.get("news") is None):
        try:
            with st.spinner("Scoring headlines with ProsusAI FinBERT..."):
                inputs["news"] = score_headlines_finbert(inputs["news"])
            st.session_state["scored_news"] = inputs["news"]
            st.sidebar.success("Headline scoring complete for this session.")
        except Exception as exc:
            st.sidebar.error(str(exc))
    if not is_demo and st.session_state.get("scored_news") is not None:
        inputs["news"] = st.session_state["scored_news"]

    try:
        frame = build_mood_frame(**inputs)
    except Exception as exc:
        st.error(f"Could not prepare these files: {exc}")
        st.stop()
    frame = frame.dropna(subset=["mood_score"]).reset_index(drop=True)
    if frame.empty:
        st.error("No dates have enough inputs to compute a mood score.")
        st.stop()
    show_data_quality(frame, is_demo)

    if reference_news is not None:
        show_reference_headline_summary(reference_news)

    if inputs.get("news") is not None:
        try:
            daily_news = aggregate_daily_sentiment(inputs["news"])
        except ValueError:
            daily_news = None
        if daily_news is not None:
            with st.expander("Daily news sentiment counts", expanded=False):
                st.dataframe(daily_news.tail(30), use_container_width=True, hide_index=True)

    now_ist = datetime.now(IST)
    can_refresh = now_ist.time() >= time(15, 30)
    latest = frame.iloc[-1]
    latest_date = pd.Timestamp(latest["Date"])
    latest_regime = str(latest["mood_regime"])

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Daily refresh**")
        st.caption("Signals use end-of-day data and target the next market session.")
        refresh = st.button("Refresh mood & direction", type="primary", disabled=not can_refresh,
                            help="Refresh is enabled after 15:30 India time.")
        if not can_refresh:
            st.caption("Available after 15:30 IST")
        regimes = st.multiselect("Regime filter", REGIME_ORDER, default=REGIME_ORDER)

    if refresh:
        try:
            with st.spinner("Fitting chronological direction model..."):
                result = train_direction_model(frame)
                probability = predict_up_probability(result["model"], frame)
            st.session_state["mood_run"] = {
                "result": result, "probability": probability,
                "computed_at": datetime.now(IST).strftime("%d %b %Y, %H:%M IST"),
            }
        except Exception as exc:
            st.warning(f"Direction model unavailable: {exc}")
            st.session_state.pop("mood_run", None)

    run = st.session_state.get("mood_run")
    st.markdown(f'<div class="timestamp">LATEST SESSION  /  {latest_date:%d %b %Y} &nbsp;&nbsp; • &nbsp;&nbsp; AS OF {now_ist:%H:%M} IST</div>', unsafe_allow_html=True)
    st.markdown("<div class='metric-rule'></div>", unsafe_allow_html=True)

    dashboard, direction, backtest = st.tabs(["Mood dashboard", "Direction signal", "Backtest"])
    with dashboard:
        left, right = st.columns([0.82, 1.45], gap="large")
        with left:
            st.plotly_chart(mood_gauge(float(latest["mood_score"]), latest_regime), use_container_width=True)
            st.markdown(f'<span class="regime-pill">{latest_regime.upper()}</span>', unsafe_allow_html=True)
            computed = run["computed_at"] if run else "Not refreshed"
            st.markdown(f'<p class="timestamp">Mood computed: {computed}</p>', unsafe_allow_html=True)
        with right:
            st.subheader("Mood components")
            component_labels = {
                "news_mood": "News tone · 30%", "volatility_mood": "Low volatility · 25%",
                "internals_mood": "Market internals · 25%", "momentum_mood": "Price momentum · 20%",
            }
            values = []
            for name, label in component_labels.items():
                value = latest.get(name, np.nan)
                values.append({"Component": label, "Score": float(value) if pd.notna(value) else np.nan})
            bars = pd.DataFrame(values).dropna()
            if not bars.empty:
                fig = go.Figure(go.Bar(x=bars["Score"], y=bars["Component"], orientation="h",
                                       marker_color=[COLORS["green"], COLORS["blue"], COLORS["gold"], COLORS["coral"]][:len(bars)],
                                       text=bars["Score"].map(lambda x: f"{x:.0f}"), textposition="outside"))
                fig.update_layout(height=230, xaxis=dict(range=[0, 110], title="Normalized component score"),
                                  yaxis=dict(autorange="reversed"), margin=dict(l=0, r=22, t=12, b=0),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            st.caption("Configured weights: " + " · ".join(f"{name.replace('_mood', '').replace('_', ' ')} {weight:.0%}" for name, weight in WEIGHTS.items()))
        visible = frame[frame["mood_regime"].isin(regimes)].tail(90)
        st.subheader("Mood and Nifty · selected sessions")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame.tail(90)["Date"], y=frame.tail(90)["mood_score"], name="Mood score (0–100)",
                                 line=dict(color=COLORS["green"], width=2.5), yaxis="y"))
        fig.add_trace(go.Scatter(x=frame.tail(90)["Date"], y=frame.tail(90)["Close"], name="Nifty close",
                                 line=dict(color=COLORS["coral"], width=1.7), yaxis="y2"))
        if len(visible) < 90:
            st.caption(f"Regime filter selects {len(visible)} of the last 90 sessions.")
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=12, b=0), hovermode="x unified",
                          yaxis=dict(title="Mood", range=[0, 100]), yaxis2=dict(title="Nifty close", overlaying="y", side="right"),
                          legend=dict(orientation="h", y=1.12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        if "mmi_score" in frame and frame["mmi_score"].notna().sum() >= 3:
            overlap = frame[["mood_score", "mmi_score"]].dropna()
            correlation = overlap["mood_score"].corr(overlap["mmi_score"])
            st.metric("Mood vs. uploaded MMI · Pearson correlation", f"{correlation:.3f}", f"{len(overlap):,} overlapping sessions")
            st.caption("Correlation is a descriptive comparison, not proof of predictive validity.")

    with direction:
        st.subheader("Next-session direction")
        if run:
            probability = run["probability"]
            signal = direction_signal(probability)
            up, signal_col, computed_col = st.columns(3)
            up.metric("P(Up)", f"{probability:.1%}")
            signal_col.metric("Model signal", signal)
            computed_col.metric("Forecast target", "Next session")
            st.caption(f"Model refreshed {run['computed_at']}. Probability thresholds: BUY >55%, SELL <45%, otherwise HOLD.")
            st.markdown("#### Chronological holdout")
            metrics = run["result"]["metrics"]
            metric_cols = st.columns(4)
            metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
            metric_cols[1].metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.1%}")
            metric_cols[2].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
            metric_cols[3].metric("Brier score", f"{metrics['brier_score']:.3f}")
            st.caption(f"{metrics['test_sessions']} out-of-time sessions · holdout predictions are separate from the final refit used for today's estimate.")
            model = run["result"]["model"]
            importance = pd.DataFrame({"Feature": model.feature_name_, "Importance": model.feature_importances_}).sort_values("Importance").tail(14)
            fig = go.Figure(go.Bar(x=importance["Importance"], y=importance["Feature"], orientation="h",
                                   marker_color=COLORS["green"]))
            fig.update_layout(height=390, margin=dict(l=0, r=12, t=10, b=0), xaxis_title="Split importance",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload enough historical sessions to train the chronological classifier. At least 120 complete labeled sessions are required.")

    with backtest:
        st.subheader("Rule-based strategy comparison")
        st.caption("Signals formed after close T earn close-to-close return on T+1. Includes 5 bps per unit of position turnover; excludes taxes, slippage, and execution constraints.")
        curve = equity_curves(backtest_rules(frame))
        fig = go.Figure()
        names = {"buy_hold": "Buy & hold", "momentum": "Momentum", "contrarian": "Contrarian", "combined": "Combined"}
        colors_by_name = {"buy_hold": COLORS["muted"], "momentum": COLORS["green"], "contrarian": COLORS["coral"], "combined": COLORS["blue"]}
        for column in ("buy_hold", "momentum", "contrarian", "combined"):
            if column in curve:
                fig.add_trace(go.Scatter(x=curve["Date"], y=curve[column], name=names[column],
                                         line=dict(color=colors_by_name[column], width=2)))
        fig.update_layout(height=400, yaxis_title="Growth of 1.0 unit", xaxis_title=None, hovermode="x unified",
                          legend=dict(orientation="h", y=1.12), margin=dict(l=0, r=0, t=14, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        ending = curve.iloc[-1]
        performance = st.columns(4)
        for column, label in zip(("buy_hold", "momentum", "contrarian", "combined"), names.values()):
            performance[list(names).index(column)].metric(label, f"{(ending[column] - 1):+.1%}")
        st.caption("Backtest results are historical diagnostics. They are not expected returns or evidence of future performance.")

    st.markdown("<div class='footer-note'>ProsusAI/finbert (optional) · Composite: 30% news / 25% volatility / 25% internals / 20% momentum · MMI correlation is shown only when uploaded · Educational research, not investment advice.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()