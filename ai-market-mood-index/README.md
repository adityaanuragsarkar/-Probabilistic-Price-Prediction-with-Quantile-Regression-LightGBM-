# AI-Driven Historical Market Mood Index

A standalone Streamlit app that combines daily financial-news sentiment, India VIX, market internals, and Nifty momentum into a 0-100 mood index. A LightGBM classifier estimates the probability that the next Nifty close is higher. Separate rule-based backtests compare momentum, contrarian, combined, and buy-and-hold returns.

> **Important:** The bundled demo data is generated and synthetic. It is visibly marked as such in the app and must not be interpreted as actual market history or a trading recommendation. No real data is downloaded automatically.

## Run locally

```bash
cd ai-market-mood-index
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

To score raw headlines with ProsusAI FinBERT, install the optional, larger dependencies:

```bash
python -m pip install -r requirements-finbert.txt
```

FinBERT downloads model weights on first use. For repeatable production runs, pin the model revision and persist the scored headlines rather than downloading weights each day.

## Data inputs

Choose **Upload historical CSVs** in the sidebar. Upload files for the sources you have. Dates are normalized and joined to Nifty trading sessions; missing sources are left missing and the mood weights are re-normalized across components actually available for that row.

- **Nifty OHLCV** (required): `Date`, `Close`; `Open`, `High`, `Low`, and `Volume` are recommended. Dates can also be named `date`, `datetime`, or `timestamp`.
- **India VIX** (optional): `Date` and `Close` (or `IndiaVIX`).
- **News** (optional): `Date` plus a `sentiment` / `label` column (`positive`, `neutral`, `negative`; or numeric labels `2`, `1`, `0`), or a numeric `sentiment_score` from -1 to 1. With the FinBERT extra installed, raw news may instead provide `headline` or `text` and use the score-headlines button.
- **Market internals** (optional): `Date`, plus any of `advance_decline_ratio`, `above_50dma_pct` (0-1 or 0-100), and `delivery_pct`.
- **TickerTape MMI** (optional): `Date` and a numeric `MMI`, `MarketMoodIndex`, `MoodIndex`, `Value`, or `Close` column. MMI is used only for historical correlation validation, not as an input feature.

The sidebar can load the public 42,214-row Hugging Face headline CSV. Its schema contains headline text and `sentiment_label` (0=negative, 1=neutral, 2=positive), but **no row-level publication date**. The app therefore displays overall label totals for this source and refuses to infer daily counts from the dataset's broad 2024-2026 date-range description. To build daily news features, supply a dated headline CSV; the app aggregates each session's positive, neutral, negative, and total counts plus mean sentiment.

The other named datasets and reference projects in the original brief are not fetched automatically. Check each provider's terms and obtain the data through its permitted interface. NSE/NSELib availability can change, and scraping or redistribution may be restricted. Add cleaned daily news labels or scores to a CSV before fitting the historical direction model.

## Method

1. Aggregate headline labels each session. A label is mapped to -1 / 0 / +1. Calculate the daily mean score, counts, five-session sentiment momentum, and sentiment volatility.
2. Convert sentiment into 0-100. Convert India VIX to an inverted trailing percentile. Build the market-internals score from available breadth/delivery fields. Map the 20-session return to a trailing historical percentile.
3. Combine components at the brief's 30% news, 25% volatility, 25% internals, and 20% price momentum weights. If a component/source is absent on a date, weights are renormalized across available components; absent data is not made up.
4. Classify score bands: Extreme Fear (<20), Fear (20-40), Neutral (40-60), Greed (60-80), and Extreme Greed (80+). K-Means validation is not included in this first version; the historical regime labels are explicit score bands.
5. Train LightGBM on causal indicators and mood features. Evaluation is a chronological 80/20 holdout, not a random split. The holdout is reported separately from the final model refit on all labeled rows.
6. Compare rule strategies on next-session close-to-close returns, with 5 basis points of cost per unit of turnover. The dashboard reports buy-and-hold beside each rule strategy.

The signal is computed from end-of-day inputs. It is intended to be reviewed after the close and before the next session. The UI disables refresh before 15:30 IST; the deployment must still schedule complete news collection, scoring, market-data availability, and refresh before displaying a final daily signal. The app displays the actual computation time instead of claiming a fixed 19:00 run.

## Guardrails and interpretation

- No cited accuracy or correlation result is assumed. The measured MMI correlation is calculated from the uploaded overlapping history and displayed with its sample size.
- A high correlation is not by itself proof of predictive value or causality.
- Direction accuracy can be close to chance. Review balanced accuracy, ROC AUC, Brier score, class balance, and the period tested.
- The toy strategy backtest excludes taxes, brokerage, slippage, intraday execution, and portfolio sizing. It is not evidence of profitability.
- FinBERT labels are model estimates, not ground truth. Check domain fit and confidence thresholds before operational use.
- Models are fit on user-provided historical files in memory. This starter project does not persist model files, schedule scraping, or place orders.

## Project layout

```text
ai-market-mood-index/
  app.py
  mood_index/
    data.py
    model.py
  tests/
    test_pipeline.py
  requirements.txt
  requirements-finbert.txt
```

## References

- TickerTape Market Mood Index: https://www.tickertape.in/market-mood-index
- Indian financial news dataset: https://huggingface.co/datasets/pranali96/indian_financial_news_42k
- Nifty 50 history dataset: https://www.kaggle.com/datasets/aparnamadathil/nifty-50-1999-2026-full-27-years-data
- NSELib: https://pypi.org/project/nselib/
- ProsusAI FinBERT: https://huggingface.co/ProsusAI/finbert
- MMI analysis reference: https://github.com/mystichronicle/market-mood-analysis
- Artha-Yukti reference: https://github.com/parth1899/Artha-Yukti