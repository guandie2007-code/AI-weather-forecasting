# AI Weather Forecasting Baseline

A machine-learning baseline for forecasting a numeric weather variable from historical observations. The pipeline creates time and lag features, uses a chronological train/test split, and compares Random Forest regression with a persistence baseline.

**Honest status:** source code only; no dataset or forecast-accuracy claim is included. Run the experiment with a documented dataset before reporting results.

## Expected CSV
Create `data/weather.csv` with timestamp and target columns, for example:

```csv
timestamp,temperature
2024-01-01 00:00:00,18.2
2024-01-01 01:00:00,17.9
2024-01-01 02:00:00,17.6
```

These values are illustrative. Your data should be real, documented, and regularly sampled for the lag features to make sense. Additional columns may be present, but this baseline currently uses the target's historical values and timestamp features.

## Setup
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/forecast.py --csv data/weather.csv --target temperature --timestamp timestamp
```

Optional:
```bash
python src/forecast.py --csv data/weather.csv --target temperature --timestamp timestamp --test-size 0.2 --lags 1 3 6 24
```

The script prints MAE and RMSE for the persistence baseline and Random Forest, and saves the model and metrics under `artifacts/`.

## Method
1. Parse and sort observations chronologically.
2. Create hour, day-of-year, month, and day-of-week features.
3. Create lag features from earlier target observations.
4. Train on earlier records and test on later records.
5. Compare a simple persistence forecast with Random Forest regression.
6. Report MAE/RMSE on the held-out time period.

## Limitations
This is a learning/research baseline, not an operational meteorological service. Data quality, location, seasonality, sampling interval, and missing values affect performance. Do not randomly shuffle time-series observations. Document data source, license, location/station, date range, and exact test period.
