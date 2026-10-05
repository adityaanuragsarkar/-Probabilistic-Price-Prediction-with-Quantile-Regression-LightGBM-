from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.feature_pipeline import create_synthetic_market_data, create_synthetic_vix_data, prepare_market_frame

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

TEAL = "#0E7C86"
UP = "#1B8A5A"
DOWN = "#C0392B"
AMBER = "#B7791F"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'DM Sans', sans-serif; }}
.block-container {{ max-width: 980px; padding-top: 2.2rem; }}
#MainMenu, footer {{ visibility: hidden; }}
.intro h1 {{ font-size: 2.1rem; font-weight: 700; margin: 0 0 .3rem 0; }}
.intro p {{ font-size: 1.05rem; opacity: .75; margin: 0 0 1.4rem 0; max-width: 620px; }}
.answer {{ padding: 1.6rem 0 .4rem 0; }}
.answer .small {{ font-size: .95rem; opacity: .7; }}
.answer .big {{ font-size: 3rem; font-weight: 700; line-height: 1.1; margin: .2rem 0; }}
.answer .change {{ font-size: 1.1rem; font-weight: 500; }}
.rb-title {{ font-weight: 700; margin: 1.8rem 0 .2rem 0; font-size: 1.1rem; }}
.rb-sub {{ opacity: .7; font-size: .95rem; margin-bottom: 1.1rem; }}
.rb-track {{ position: relative; height: 14px; border-radius: 7px; background: rgba(127,127,127,.18); margin: 1rem 0 1.1rem 0; }}
.rb-band {{ position: absolute; top: 0; height: 14px; border-radius: 7px; background: {TEAL}; opacity: .35; }}
.rb-dot {{ position: absolute; top: -5px; width: 24px; height: 24px; margin-left: -12px; border-radius: 50%; border: 3px solid var(--background-color, #fff); box-sizing: border-box; }}
.rb-dot.today {{ background: #7f8c8d; }}
.rb-dot.exp {{ background: {TEAL}; }}
.rb-legend {{ display: flex; flex-wrap: wrap; gap: 1.4rem 2.2rem; font-size: .95rem; }}
.rb-legend b {{ display: block; font-size: 1.15rem; }}
.rb-legend span.k {{ opacity: .7; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1.8rem 0 1rem 0; }}
.card {{ border: 1px solid rgba(127,127,127,.25); border-radius: 12px; padding: 1.1rem 1.2rem; border-top: 5px solid var(--c); }}
.card .q {{ font-size: .9rem; opacity: .7; }}
.card .a {{ font-size: 1.5rem; font-weight: 700; margin: .25rem 0 .35rem 0; color: var(--c); }}
.card .why {{ font-size: .95rem; opacity: .85; line-height: 1.45; }}
.fine {{ font-size: .85rem; opacity: .65; margin-top: 1.6rem; line-height: 1.5; }}
</style>
"""


@st.cache_resource
def load_models():
    required = [
        MODELS_DIR / "lgbm_p10.pkl",
        MODELS_DIR / "lgbm_p50.pkl",
        MODELS_DIR / "lgbm_p90.pkl",
        MODELS_DIR / "feature_columns.pkl",
        MODELS_DIR / "conformity_scores.pkl",
        MODELS_DIR / "interval_widths.pkl",
    ]
    if not all(path.exists() for path in required):
        from src.train_quantile_model import main as train_main

        train_main()

    with open(MODELS_DIR / "lgbm_p10.pkl", "rb") as f:
        model_p10 = pickle.load(f)
    with open(MODELS_DIR / "lgbm_p50.pkl", "rb") as f:
        model_p50 = pickle.load(f)
    with open(MODELS_DIR / "lgbm_p90.pkl", "rb") as f:
        model_p90 = pickle.load(f)
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open(MODELS_DIR / "conformity_scores.pkl", "rb") as f:
        conformity_scores = pickle.load(f)
    with open(MODELS_DIR / "interval_widths.pkl", "rb") as f:
        interval_widths = pickle.load(f)

    return model_p10, model_p50, model_p90, feature_columns, conformity_scores, interval_widths


@st.cache_data(ttl=3600)
def load_latest_data():
    synthetic_used = False
    nifty_path = DATA_DIR / "nifty50.csv"
    vix_path = DATA_DIR / "india_vix.csv"

    if nifty_path.exists():
        nifty_df = pd.read_csv(nifty_path)
    else:
        nifty_df = create_synthetic_market_data(n_rows=2200, start_date="2015-01-01")
        synthetic_used = True  # Synthetic data stays in memory so this warning repeats.

    if vix_path.exists():
        vix_df = pd.read_csv(vix_path)
    else:
        vix_df = create_synthetic_vix_data(n_rows=2200, start_date="2015-01-01")
        synthetic_used = True

    df = prepare_market_frame(nifty_df, vix_df)
    df = df.sort_values("Date").reset_index(drop=True)
    return df, synthetic_used


def calculate_position_size(interval_width: float, historical_widths: np.ndarray) -> float:
    q30 = np.quantile(historical_widths, 0.30)
    q70 = np.quantile(historical_widths, 0.70)

    if interval_width <= q30:
        return 1.0
    if interval_width <= q70:
        return 0.75
    return 0.50


def compute_signal(predicted_close: float, current_close: float) -> str:
    if predicted_close > current_close * 1.005:
        return "BUY"
    if predicted_close < current_close * 0.995:
        return "SELL"
    return "HOLD"


def money(x: float) -> str:
    return f"₹{x:,.0f}"


def show_chart(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def range_bar(lower: float, upper: float, current: float, expected: float) -> str:
    lo, hi = min(lower, current), max(upper, current)
    pad = (hi - lo) * 0.15
    lo, hi = lo - pad, hi + pad

    def pos(v: float) -> float:
        return (v - lo) / (hi - lo) * 100

    return (
        '<div class="rb-title">Where tomorrow\'s price will most likely land</div>'
        '<div class="rb-sub">The shaded stretch is the likely range. '
        'In past tests, the real price landed inside a range like this about 4 times out of 5.</div>'
        '<div class="rb-track">'
        f'<div class="rb-band" style="left:{pos(lower):.1f}%;width:{pos(upper) - pos(lower):.1f}%"></div>'
        f'<div class="rb-dot today" style="left:{pos(current):.1f}%"></div>'
        f'<div class="rb-dot exp" style="left:{pos(expected):.1f}%"></div>'
        '</div>'
        '<div class="rb-legend">'
        f'<div><span class="k">Lowest likely</span><b>{money(lower)}</b></div>'
        f'<div><span class="k">Today (grey dot)</span><b>{money(current)}</b></div>'
        f'<div><span class="k">Expected (teal dot)</span><b>{money(expected)}</b></div>'
        f'<div><span class="k">Highest likely</span><b>{money(upper)}</b></div>'
        '</div>'
    )


def card(question: str, answer: str, why: str, color: str) -> str:
    return (
        f'<div class="card" style="--c:{color}">'
        f'<div class="q">{question}</div><div class="a">{answer}</div><div class="why">{why}</div></div>'
    )


SIGNAL_TEXT = {
    "BUY": ("Leaning up", UP, "The model expects the market to rise by more than half a percent."),
    "SELL": ("Leaning down", DOWN, "The model expects the market to fall by more than half a percent."),
    "HOLD": ("No clear direction", AMBER, "The expected move is small, so the model sees no strong reason to act."),
}

SIZE_TEXT = {
    1.0: ("Normal amount", UP, "The market looks calm and the forecast range is narrow, so the model is fairly sure."),
    0.75: ("Slightly less than usual", AMBER, "There is some uncertainty, so it suggests going a little lighter."),
    0.50: ("Half of usual", DOWN, "The forecast range is wide, meaning the model is unsure. It suggests being careful."),
}


def render_dashboard():
    st.set_page_config(page_title="Nifty 50 Forecast", page_icon="📈", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="intro"><h1>Nifty 50: tomorrow\'s forecast</h1>'
        "<p>The Nifty 50 is an index that tracks India's 50 largest listed companies. "
        "This tool looks at past market behaviour and estimates where the index may close next trading day.</p></div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Getting the model ready..."):
        model_p10, model_p50, model_p90, feature_columns, conformity_scores, interval_widths = load_models()
        market_df, synthetic_used = load_latest_data()

    if synthetic_used:
        st.warning("These numbers use made-up practice data, because the real market files were not found. Do not rely on them.")

    latest_data_date = pd.to_datetime(market_df["Date"].iloc[-1]).normalize()
    if latest_data_date < pd.Timestamp.now().normalize() - pd.Timedelta(days=7):
        st.warning(
            f"Market data is stale. The latest available close is {latest_data_date:%d %b %Y}; "
            "this forecast is based on that date, not today's market."
        )

    use_sizing = st.toggle("Also suggest how much to invest", value=True)
    if st.button("Show tomorrow's forecast", type="primary"):
        st.session_state["ran"] = True

    if not st.session_state.get("ran"):
        st.info("Click the button above after the market closes to see the forecast.")
        return

    feature_df = (
        market_df[feature_columns].replace([np.inf, -np.inf], np.nan).ffill().tail(1).copy()
    )
    p10 = float(model_p10.predict(feature_df)[0])
    p50 = float(model_p50.predict(feature_df)[0])
    p90 = float(model_p90.predict(feature_df)[0])
    p10, p50, p90 = sorted([p10, p50, p90])

    q = float(np.quantile(conformity_scores, 0.80))
    lower, upper = p10 - q, p90 + q
    current_close = float(market_df.iloc[-1]["Close"])
    signal = compute_signal(p50, current_close)
    size = calculate_position_size(upper - lower, interval_widths)
    change = p50 / current_close - 1

    last_date = latest_data_date
    next_date = pd.bdate_range(start=last_date, periods=2)[-1]

    color = UP if change > 0.0005 else DOWN if change < -0.0005 else AMBER
    arrow = "▲" if change > 0.0005 else "▼" if change < -0.0005 else "■"
    st.markdown(
        f'<div class="answer"><div class="small">Expected close on {next_date.strftime("%A, %d %b %Y")}</div>'
        f'<div class="big">{money(p50)}</div>'
        f'<div class="change" style="color:{color}">{arrow} {change:+.2%} compared with today\'s close of {money(current_close)}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(range_bar(lower, upper, current_close, p50), unsafe_allow_html=True)

    s_title, s_color, s_why = SIGNAL_TEXT[signal]
    z_title, z_color, z_why = SIZE_TEXT[size]
    if not use_sizing:
        z_title, z_color, z_why = "Not shown", "#7f8c8d", "You turned off investment-size suggestions above."
    width_pct = (upper - lower) / current_close
    sure_title, sure_color = (
        ("Fairly sure", UP) if size == 1.0 else ("Somewhat unsure", AMBER) if size == 0.75 else ("Quite unsure", DOWN)
    )
    st.markdown(
        '<div class="cards">'
        + card("Which way is it leaning?", s_title, s_why, s_color)
        + card("How sure is the model?", sure_title,
               f"The likely range spans about {width_pct:.1%} of today's price. A narrower range means more confidence.", sure_color)
        + card("Suggested amount to invest", z_title, z_why, z_color)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rb-title">The last 200 trading days</div>', unsafe_allow_html=True)
    chart_df = market_df.tail(200).copy()
    chart_df["Date"] = pd.to_datetime(chart_df["Date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["Date"], y=chart_df["Close"], mode="lines",
                             name="Past closing price", line=dict(color="#7f8c8d", width=2)))
    fig.add_trace(go.Scatter(
        x=[last_date, next_date, next_date, last_date],
        y=[current_close, upper, lower, current_close],
        fill="toself", mode="none", name="Likely range",
        fillcolor="rgba(14,124,134,0.25)",
    ))
    fig.add_trace(go.Scatter(x=[last_date, next_date], y=[current_close, p50], mode="lines+markers",
                             name="Forecast", line=dict(color=TEAL, width=3, dash="dot"),
                             marker=dict(size=[0, 12], color=TEAL)))
    fig.update_layout(
        template="plotly_white", height=380, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Index level (₹)", xaxis_title=None,
        legend=dict(orientation="h", y=1.1), hovermode="x unified",
    )
    show_chart(fig)

    with st.expander("How do I read this?"):
        st.markdown(
            "- **Expected close** is the model's single best guess for the next trading day.\n"
            "- **Likely range** is wider on purpose: no one can predict the market exactly, so the tool shows where the price will probably land, not just one number.\n"
            "- **Leaning up / down** only appears when the expected move is bigger than half a percent.\n"
            "- **Suggested amount** goes down when the range gets wider, because a wider range means the model is less sure."
        )
    with st.expander("For technical readers"):
        st.write(
            f"Median (p50) prediction: {p50:,.2f}. Raw quantile band: {p10:,.2f} to {p90:,.2f}. "
            f"Conformal adjustment: ±{q:,.2f} to target 80% coverage. Interval width: {upper - lower:,.2f} "
            f"(historical range {np.min(interval_widths):,.2f} to {np.max(interval_widths):,.2f})."
        )

    st.markdown(
        '<div class="fine">This is a learning project, not financial advice. Forecasts can be wrong, '
        "and you can lose money if you act on them.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_dashboard()
