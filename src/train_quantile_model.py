from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from feature_pipeline import (
    build_feature_matrix,
    create_synthetic_market_data,
    create_synthetic_vix_data,
    prepare_market_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


def try_load_vix_from_nselib() -> pd.DataFrame | None:
    try:
        import nselib  # noqa: F401

        try:
            from nselib import equity  # type: ignore
        except Exception:
            equity = None

        if equity is not None and hasattr(equity, "india_vix_data"):
            df = equity.india_vix_data()
            if isinstance(df, pd.DataFrame):
                return df

        if hasattr(nselib, "india_vix_data"):
            df = nselib.india_vix_data()
            if isinstance(df, pd.DataFrame):
                return df
    except Exception:
        return None
    return None


def ensure_artifacts_exist() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    nifty_path = DATA_DIR / "nifty50.csv"
    vix_path = DATA_DIR / "india_vix.csv"

    if nifty_path.exists():
        nifty_df = pd.read_csv(nifty_path)
    else:
        nifty_df = create_synthetic_market_data(n_rows=2200, start_date="2015-01-01")
        nifty_df.to_csv(nifty_path, index=False)

    if vix_path.exists():
        vix_df = pd.read_csv(vix_path)
    else:
        vix_df = try_load_vix_from_nselib()
        if vix_df is None:
            vix_df = create_synthetic_vix_data(n_rows=2200, start_date="2015-01-01")
        vix_df.to_csv(vix_path, index=False)

    return nifty_df, vix_df


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        train_df = df[df["Date"] < "2018-01-01"].copy()
        val_df = df[(df["Date"] >= "2018-01-01") & (df["Date"] < "2023-01-01")].copy()
        test_df = df[df["Date"] >= "2023-01-01"].copy()
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            total = len(df)
            train_cut = int(total * 0.7)
            val_cut = int(total * 0.9)
            train_df = df.iloc[:train_cut].copy()
            val_df = df.iloc[train_cut:val_cut].copy()
            test_df = df.iloc[val_cut:].copy()
        return train_df, val_df, test_df

    total = len(df)
    train_cut = int(total * 0.7)
    val_cut = int(total * 0.9)
    return df.iloc[:train_cut], df.iloc[train_cut:val_cut], df.iloc[val_cut:]


def compute_conformal_scores(p10: np.ndarray, p90: np.ndarray, actual: np.ndarray) -> np.ndarray:
    scores = np.maximum(actual - p90, p10 - actual)
    return np.clip(scores, 0, None)


def sort_quantiles(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.column_stack([np.asarray(p10), np.asarray(p50), np.asarray(p90)])
    sorted_matrix = np.sort(matrix, axis=1)
    return sorted_matrix[:, 0], sorted_matrix[:, 1], sorted_matrix[:, 2]


def train_quantile_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, alpha: float):
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        learning_rate=0.05,
        n_estimators=500,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def evaluate_predictions(actual: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray):
    mae = float(np.mean(np.abs(actual - p50)))
    rmse = float(np.sqrt(np.mean((actual - p50) ** 2)))
    mape = float(np.mean(np.abs((actual - p50) / np.maximum(actual, 1e-8))) * 100)

    lower = p10
    upper = p90
    coverage = float(np.mean((actual >= lower) & (actual <= upper))) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Coverage": coverage,
    }


def main() -> dict:
    nifty_df, vix_df = ensure_artifacts_exist()
    processed = prepare_market_frame(nifty_df, vix_df)
    train_df, val_df, test_df = split_by_time(processed)

    X_train, y_train, _ = build_feature_matrix(train_df)
    X_val, y_val, _ = build_feature_matrix(val_df)
    X_test, y_test, feature_cols = build_feature_matrix(test_df)

    model_p10 = train_quantile_model(X_train, y_train, X_val, y_val, alpha=0.10)
    model_p50 = train_quantile_model(X_train, y_train, X_val, y_val, alpha=0.50)
    model_p90 = train_quantile_model(X_train, y_train, X_val, y_val, alpha=0.90)

    val_p10 = model_p10.predict(X_val)
    val_p50 = model_p50.predict(X_val)
    val_p90 = model_p90.predict(X_val)
    val_p10, val_p50, val_p90 = sort_quantiles(val_p10, val_p50, val_p90)
    val_scores = compute_conformal_scores(val_p10, val_p90, y_val.to_numpy())
    conformal_q = float(np.quantile(val_scores, 0.80))
    val_widths = (val_p90 - val_p10) + 2 * conformal_q

    test_p10 = model_p10.predict(X_test)
    test_p50 = model_p50.predict(X_test)
    test_p90 = model_p90.predict(X_test)
    test_p10, test_p50, test_p90 = sort_quantiles(test_p10, test_p50, test_p90)
    test_lower = test_p10 - conformal_q
    test_upper = test_p90 + conformal_q

    metrics = evaluate_predictions(y_test.to_numpy(), test_lower, test_p50, test_upper)

    with open(MODELS_DIR / "lgbm_p10.pkl", "wb") as f:
        pickle.dump(model_p10, f)
    with open(MODELS_DIR / "lgbm_p50.pkl", "wb") as f:
        pickle.dump(model_p50, f)
    with open(MODELS_DIR / "lgbm_p90.pkl", "wb") as f:
        pickle.dump(model_p90, f)
    with open(MODELS_DIR / "feature_columns.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    with open(MODELS_DIR / "conformity_scores.pkl", "wb") as f:
        pickle.dump(np.array(val_scores), f)
    with open(MODELS_DIR / "interval_widths.pkl", "wb") as f:
        pickle.dump(np.array(val_widths), f)

    print("Quantile regression training complete.")
    print(metrics)
    return {
        "metrics": metrics,
        "conformal_q": conformal_q,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    main()
