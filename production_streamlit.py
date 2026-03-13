from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from production_DL_pipeline import (
	MODEL_PATH,
	SCALER_PATH,
	load_data_df,
	predict_with_lstm,
	retrain_lstm_from_df,
)


st.set_page_config(page_title="LSTM Production Pipeline", layout="wide")
st.title("LSTM Production Pipeline")
st.caption("Load data, run multiday prediction, or trigger retraining from the Streamlit UI.")

if "loaded_data_df" not in st.session_state:
	st.session_state.loaded_data_df = None
if "loaded_data_info" not in st.session_state:
	st.session_state.loaded_data_info = ""


def _build_next_days_forecast(predictions_df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
	"""Create a compact next-days forecast table from the latest prediction row."""
	if predictions_df.empty:
		raise ValueError("Predictions dataframe is empty")

	latest_prediction = predictions_df.iloc[-1]
	if reference_date is not None:
		last_timestamp = pd.Timestamp(reference_date)
	else:
		last_timestamp = pd.to_datetime(predictions_df.index[-1])

	forecast_rows = []
	for column_name, value in latest_prediction.items():
		match = re.search(r"t\+(\d+)", str(column_name))
		if match is None:
			continue
		horizon = int(match.group(1))
		forecast_rows.append(
			{
				"target": column_name,
				"horizon_days": horizon,
				"forecast_date": last_timestamp + pd.Timedelta(days=horizon),
				"predicted_load_mw": float(value),
			}
		)

	if not forecast_rows:
		raise ValueError("No target horizon columns found (expected names like load_t+1)")

	forecast_df = pd.DataFrame(forecast_rows).sort_values("horizon_days").reset_index(drop=True)
	return forecast_df


def _get_reference_date(df: pd.DataFrame) -> pd.Timestamp | None:
	if "time" not in df.columns:
		return None
	time_series = pd.Series(pd.to_datetime(df["time"], errors="coerce")).dropna()
	if time_series.empty:
		return None
	if getattr(time_series.dt, "tz", None) is not None:
		time_series = time_series.dt.tz_convert(None)
	return pd.Timestamp(time_series.max()).normalize()


def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
	suffix = Path(uploaded_file.name).suffix.lower()
	if suffix == ".csv":
		df = pd.read_csv(uploaded_file)
	elif suffix == ".parquet":
		df = pd.read_parquet(uploaded_file)
	else:
		raise ValueError("Only .csv and .parquet uploads are supported")

	if "time" in df.columns:
		df["time"] = pd.to_datetime(df["time"])
	return df


st.sidebar.header("Pipeline Settings")
action = st.sidebar.selectbox("Action", ["predict", "retrain"])
data_source = st.sidebar.radio("Data source", ["Upload file", "Fetch automatically"])

model_path = st.sidebar.text_input("Model path", value=str(MODEL_PATH))
scaler_path = st.sidebar.text_input("Scaler path", value=str(SCALER_PATH))

data_df = st.session_state.loaded_data_df
uploaded = None
start_date = date(2018, 1, 1)
end_date = None
country_code = "DE"
locations = 3
api_key_input = ""

if data_source == "Upload file":
	uploaded = st.sidebar.file_uploader("Upload .csv or .parquet", type=["csv", "parquet"])
else:
	start_date = st.sidebar.date_input("Start date", value=date(2018, 1, 1))
	end_date_option = st.sidebar.checkbox("Set custom end date", value=False)
	if end_date_option:
		end_date = st.sidebar.date_input("End date", value=date.today() - timedelta(days=1))
	else:
		end_date = None
		st.sidebar.caption("End date will default to yesterday automatically.")

	country_code = st.sidebar.text_input("Country code", value="DE")
	api_key_input = st.sidebar.text_input("ENTSOE API key (optional)", value="", type="password")
	locations = st.sidebar.number_input("Number of locations", min_value=1, max_value=5, value=3, step=1)

if st.sidebar.button("Load data"):
	with st.spinner("Loading data..."):
		try:
			if data_source == "Upload file":
				if uploaded is None:
					raise ValueError("Please choose a file to upload first.")
				loaded_df = _read_uploaded_file(uploaded)
				st.session_state.loaded_data_df = loaded_df
				st.session_state.loaded_data_info = f"Uploaded file: {uploaded.name}"
				data_df = loaded_df
				st.success(f"Loaded {len(data_df)} rows from {uploaded.name}")
			else:
				loaded_df = load_data_df(
					data_path=None,
					start_date=start_date,
					end_date=end_date,
					country_code=country_code,
					locations=int(locations),
					api_key=api_key_input.strip() or None,
				)
				st.session_state.loaded_data_df = loaded_df
				resolved_end = pd.Timestamp(end_date).date() if end_date is not None else (date.today() - timedelta(days=1))
				st.session_state.loaded_data_info = f"Fetched data: {start_date} to {resolved_end}"
				data_df = loaded_df
				st.success(f"Fetched {len(data_df)} rows")
		except Exception as exc:
			st.error(f"Failed to load data: {exc}")

if st.sidebar.button("Clear loaded data"):
	st.session_state.loaded_data_df = None
	st.session_state.loaded_data_info = ""
	data_df = None

if data_df is not None:
	st.subheader("Loaded Data")
	if st.session_state.loaded_data_info:
		st.caption(st.session_state.loaded_data_info)
	st.dataframe(data_df.tail(20), use_container_width=True)
	st.write(f"Rows: {len(data_df)}")

st.markdown("---")

validation_fraction = 0.2
validation_start_date = None

if action == "retrain":
	st.subheader("Retraining Configuration")
	validation_mode = st.radio("Validation split mode", ["Fraction", "Start date"])

	if validation_mode == "Fraction":
		validation_fraction = st.slider("Validation fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
	else:
		validation_start_date = st.date_input("Validation start date", value=date(2025, 1, 1))

if st.button("Run"):
	if data_df is None:
		st.error("Please load data first.")
	else:
		try:
			if action == "predict":
				with st.spinner("Running prediction..."):
					predictions_df = predict_with_lstm(
						data_df=data_df,
						model_path=model_path,
						scaler_path=scaler_path,
					)

				st.success("Prediction finished.")
				st.subheader("Predictions")
				st.dataframe(predictions_df.tail(30), use_container_width=True)

				st.subheader("Prediction Plot")
				st.line_chart(predictions_df)

				st.subheader("Next Days Forecast")
				reference_date = _get_reference_date(data_df)
				next_days_df = _build_next_days_forecast(predictions_df, reference_date=reference_date)
				st.dataframe(next_days_df, use_container_width=True)
				st.line_chart(next_days_df.set_index("forecast_date")[["predicted_load_mw"]])

			else:
				with st.spinner("Retraining model from scratch..."):
					result = retrain_lstm_from_df(
						data_df=data_df,
						validation_fraction=validation_fraction,
						validation_start_date=str(validation_start_date) if validation_start_date is not None else None,
						checkpoint_path=model_path,
						scaler_path=scaler_path,
					)

				st.success("Retraining finished.")
				st.json(
					{
						"checkpoint_path": result["checkpoint_path"],
						"scaler_path": result["scaler_path"],
						"best_epoch": result["best_epoch"],
						"best_val_mse": result["best_val_mse"],
					}
				)

				with st.spinner("Generating forecast from freshly trained model..."):
					predictions_df = predict_with_lstm(
						data_df=data_df,
						model_path=model_path,
						scaler_path=scaler_path,
					)

				st.subheader("Next Days Forecast (Freshly Trained Model)")
				reference_date = _get_reference_date(data_df)
				next_days_df = _build_next_days_forecast(predictions_df, reference_date=reference_date)
				st.dataframe(next_days_df, use_container_width=True)
				st.line_chart(next_days_df.set_index("forecast_date")[["predicted_load_mw"]])

		except Exception as exc:
			st.error(f"Pipeline failed: {exc}")
