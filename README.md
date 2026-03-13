# Time Series Energy Prediction

Forecast day-ahead electricity load using weather, calendar context, and historical load signals.

This repository combines:
- Data collection from ENTSO-E (load) and Meteostat (weather)
- Feature engineering for classical ML and deep learning
- Model training notebooks for Random Forest, XGBoost, and LSTM

## What This Project Does

The pipeline builds a daily dataset and predicts next-day load.

1. Pull load time series from ENTSO-E.
2. Pull weather for multiple German cities and average to national-level weather indicators.
3. Engineer lag, rolling, and calendar features.
4. Train and evaluate models on chronological splits:
	- Train: 2018-2023
	- Validation: 2024
	- Test: 2025

## Repository Structure

- data/ : cached parquet splits and artifacts
- models/ : trained model files and scaler
- utils/get_features.py : data acquisition and feature engineering
- utils/data_preparation.py : split, scaling, dataset preparation
- utils/visualize_model_performance.py : evaluation plots and metrics
- training_ML_models.ipynb : Random Forest and XGBoost workflow
- training_DL_models.ipynb : LSTM workflow

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Lightweight install hint

The current requirements.txt includes many notebook-related packages because it was generated from a full environment snapshot.

Optional: install core packages manually:

```bash
pip install pandas numpy scikit-learn xgboost torch meteostat entsoe-py holidays python-dotenv matplotlib seaborn joblib fastparquet
```

Then add any missing package if your notebook reports ImportError.

## API Key Setup

This project expects an ENTSO-E API key in an environment variable.

Create a .env file in the project root:

```env
ENTSOE_API_KEY=your_key_here
```

Notes:
- The loader searches for .env in the project root and current working directory.
- You can also pass api_key directly to data functions.

## Feature Engineering (Current)

### Load features

Generated in utils/get_features.py:

- load: daily average load for current day
- load_lag_1: previous day load (shifted by 1)
- load_lag_7: target-day-aligned weekly lag (shifted by 6)
- load_lag_14: target-day-aligned biweekly lag (shifted by 13)
- load_lag_30: target-day-aligned monthly lag (shifted by 29)
- rolling_mean_7, rolling_mean_14, rolling_mean_30
- std_7, std_14, std_30
- load_t+1: target, next-day load

### Weather features

Averaged across selected cities (configured in the notebook call):

- Temp
- Min Temp
- Max Temp
- Wind Speed
- Sunshine Duration
- Cloud Cover
- Cooling Degrees: degrees above 22 °C (cooling demand proxy)
- Heating Degrees: degrees below 17 °C (heating demand proxy)

### Calendar features

- is_holiday: Germany-wide holidays
- dow_sin, dow_cos: Cyclical encoded day of the week
- month_sin, month_cos: Cyclical encoded month

When align_calendar_to_target_day=True (default in the matching function), calendar features are shifted so row t uses calendar context for target day t+1.

### Feature Importance Analysis

Feature importance was estimated using permutation importance with the trained Random Forest forecaster on the 2025 test dataset. Features with larger performance drops are more influential for day-ahead load prediction.

![Random Forest permutation feature importance](img/feature_importance.png)

The SHAP feature importance analysis provides quite similar insights:

![SHAP feature importance](img/shap_feature_importance.png)

### Omitted Features

Based on permutation importance, XGBoost feature importance, and SHAP analysis, the following features contribute negligibly to predictive performance and are excluded from model training (`LOW_IMPORTANCE_FEATURES`):

| Feature | Reason |
|---|---|
| Wind Speed | Near-zero importance across all methods |
| Sunshine Duration | Near-zero importance across all methods |
| Cloud Cover | Near-zero importance across all methods |
| Cooling Degrees | Minimal contribution; dominated by temperature features |
| Heating Degrees | Minimal contribution; dominated by temperature features |
| std_7 | Rolling standard deviation adds no predictive signal |
| std_14 | Rolling standard deviation adds no predictive signal |

The active feature set used for training is: `Temp`, `Min Temp`, `Max Temp`, `load`, `load_lag_1`, `load_lag_7`, `load_lag_14`, `rolling_mean_7`, `rolling_mean_14`, `is_holiday`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`.


## Data Preparation

utils/data_preparation.py does the following:

- Loads or regenerates train/validation/test parquet files
- Builds feature matrix and target vector from selected columns
- Fits StandardScaler only on training features listed in SCALE_FEATURES
- Applies the same scaler to validation and test
- Saves scaler to models/feature_scaler.joblib

## Training Workflows

### Classical ML

Open training_ML_models.ipynb and run cells in order.

Models:
- RandomForestRegressor
- XGBRegressor

### Deep Learning

Open training_DL_models.ipynb and run cells in order.

Model:
- LSTMForecaster (windowed sequence input)

## Results

### Random Forest Forecaster
This plot shows the predicted next-day load for the 2025 test dataset using the trained Random Forest forecaster:

![Random Forest test dataset output](img/output_rf_test_ds.png)

### XGBoost Forecaster
This plot shows the predicted next-day load for the 2025 test dataset using the trained XGBoost forecaster:

![XGBoost test dataset output](img/output_xgboost_test_ds.png)

### LSTM Forecaster
This plot shows the predicted next-day load for the 2025 test dataset using the trained LSTM forecaster:

![LSTM test dataset output](img/output_lstm_test_ds.png)

### Performance Comparison (Test dataset; 2025)

| Model | MAE | RMSE |
|---|---|---|
| Random Forest | 1044.60 | 1365.51 |
| XGBoost | 975.32 | 1286.03 |
| LSTM |  960.19 | 1232.26 |

### Takeaway
The LSTM model sligtly outperforms the other models and therefore will be used for the deployment in the Streamlit application and the multi-day forecasting.

