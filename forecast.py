"""Train and evaluate a weather forecasting baseline using chronological data."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="temperature")
    ap.add_argument("--timestamp", default="timestamp")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 3, 6, 24])
    ap.add_argument("--output-dir", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    path = Path(a.csv)
    if not path.is_file():
        raise SystemExit(f"CSV file not found: {path}")
    if not 0 < a.test_size < 0.5:
        raise SystemExit("--test-size must be between 0 and 0.5.")
    raw = pd.read_csv(path)
    for col in (a.timestamp, a.target):
        if col not in raw.columns:
            raise SystemExit(f"Required column {col!r} missing; available columns: {list(raw.columns)}")
    raw[a.timestamp] = pd.to_datetime(raw[a.timestamp], errors="coerce")
    raw[a.target] = pd.to_numeric(raw[a.target], errors="coerce")
    raw = raw.dropna(subset=[a.timestamp, a.target]).sort_values(a.timestamp)
    raw = raw.drop_duplicates(subset=[a.timestamp], keep="last")
    feat = pd.DataFrame(index=raw.index)
    ts = raw[a.timestamp]
    feat["hour"] = ts.dt.hour
    feat["day_of_year"] = ts.dt.dayofyear
    feat["month"] = ts.dt.month
    feat["day_of_week"] = ts.dt.dayofweek
    for lag in sorted(set(a.lags)):
        if lag < 1:
            raise SystemExit("All lags must be positive integers.")
        feat[f"lag_{lag}"] = raw[a.target].shift(lag)
    dataset = feat.copy()
    dataset["target"] = raw[a.target]
    dataset = dataset.dropna()
    if len(dataset) < 30:
        raise SystemExit(f"Only {len(dataset)} usable rows; at least 30 are recommended.")
    cols = [c for c in dataset.columns if c != "target"]
    split = int(len(dataset) * (1 - a.test_size))
    train, test = dataset.iloc[:split], dataset.iloc[split:]
    if len(train) < 10 or len(test) < 5:
        raise SystemExit("Not enough rows for a meaningful chronological split.")
    model = RandomForestRegressor(n_estimators=300, random_state=a.seed,
                                  min_samples_leaf=2, n_jobs=-1)
    model.fit(train[cols], train["target"])
    pred = model.predict(test[cols])
    baseline = test["lag_1"].to_numpy()
    metrics = {
        "target": a.target, "usable_rows": int(len(dataset)),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "test_start": str(raw.loc[test.index, a.timestamp].min()),
        "test_end": str(raw.loc[test.index, a.timestamp].max()),
        "persistence_baseline": {
            "mae": float(mean_absolute_error(test["target"], baseline)),
            "rmse": float(np.sqrt(mean_squared_error(test["target"], baseline)))},
        "random_forest": {
            "mae": float(mean_absolute_error(test["target"], pred)),
            "rmse": float(np.sqrt(mean_squared_error(test["target"], pred)))},
        "features": cols, "lags": sorted(set(a.lags)), "seed": a.seed}
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": cols, "target": a.target},
                out / "weather_model.joblib")
    (out / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {out / 'weather_model.joblib'}")
    print(f"Saved evaluation: {out / 'evaluation.json'}")

if __name__ == "__main__":
    main()
