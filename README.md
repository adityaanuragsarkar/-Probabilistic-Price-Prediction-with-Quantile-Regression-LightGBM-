# Probabilistic Price Prediction with Quantile Regression (LightGBM)

This project trains three LightGBM quantile models to estimate the 10th, 50th, and 90th percentiles of tomorrow's Nifty 50 close and exposes the result in a Streamlit app.

## Project structure

- `src/feature_pipeline.py` – feature engineering and synthetic-data fallback
- `src/train_quantile_model.py` – training, calibration, and model serialization
- `app.py` – Streamlit dashboard for inference and risk-aware position sizing
- `requirements.txt` – Python dependencies

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/train_quantile_model.py
streamlit run app.py
```

## Data

The training workflow expects daily Nifty data plus India VIX data if available. The project attempts to use `nselib` for India VIX ingestion when it's installed, and otherwise falls back to a synthetic dataset so the app can still run in a demo environment.

## Model logic

- `p10` model: lower quantile
- `p50` model: median / point forecast
- `p90` model: upper quantile
- conformal calibration: interval width is widened using validation-set conformity scores
- position sizing: based on the current 80% interval width relative to historical widths

## Notes

This is designed for a daily after-close signal generation workflow and uses only information available up to the close of day T.
