from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from utils.get_features import get_matched_weather_load_data
from utils.lstm_model import LSTMForecaster
from utils.train_lstm import FEATURES, TARGETS, WINDOW_SIZE, DEVICE, train_lstm_model


MODEL_PATH = Path("models/best_lstm_multiday_load_forecaster.pt")
SCALER_PATH = Path("models/best_lstm_multiday_feature_scaler.joblib")


def load_data_df(
    data_path: str | Path | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    country_code: str = "DE",
    locations: int = 3,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Load raw dataframe either from disk or by fetching fresh data."""
    if data_path is not None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if data_path.suffix == ".parquet":
            data_df = pd.read_parquet(data_path)
        elif data_path.suffix == ".csv":
            data_df = pd.read_csv(data_path)
        else:
            raise ValueError("Only .parquet and .csv input files are supported")
    else:
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required when data_path is not provided")

        resolved_start = pd.Timestamp(start_date).date()
        resolved_end = pd.Timestamp(end_date).date()
        data_df = get_matched_weather_load_data(
            start_date=resolved_start,
            end_date=resolved_end,
            country_code=country_code,
            locations=locations,
            api_key=api_key,
            align_calendar_to_target_day=True,
        )

    if "time" in data_df.columns:
        data_df["time"] = pd.to_datetime(data_df["time"])

    return data_df


def load_lstm_model(
    model_path: str | Path = MODEL_PATH,
    device: torch.device = DEVICE,
) -> tuple[LSTMForecaster, dict]:
    """Load a trained multiday LSTM checkpoint."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if not (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
        raise RuntimeError("Checkpoint format is incompatible")

    features = checkpoint.get("features", FEATURES)
    targets = checkpoint.get("targets", TARGETS)

    model = LSTMForecaster(
        input_size=len(features),
        hidden_size=64,
        num_layers=1,
        output_size=len(targets),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def _build_feature_windows(X: np.ndarray, window_size: int) -> torch.Tensor:
    X_np = np.asarray(X)
    if len(X_np) < window_size:
        raise ValueError("Not enough rows in data_df for the configured window_size")

    X_windows = []
    for idx in range(len(X_np) - window_size + 1):
        X_windows.append(X_np[idx:idx + window_size])

    return torch.tensor(np.asarray(X_windows), dtype=torch.float32)


def predict_with_lstm(
    data_df: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    scaler_path: str | Path = SCALER_PATH,
    device: torch.device = DEVICE,
) -> pd.DataFrame:
    """Load a saved model and scaler, then generate multi-target predictions from raw input data."""
    model, checkpoint = load_lstm_model(model_path=model_path, device=device)

    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Feature scaler not found: {scaler_path}. Retrain the model so the scaler is saved."
        )

    scaler = joblib.load(scaler_path)
    features = checkpoint.get("features", FEATURES)
    targets = checkpoint.get("targets", TARGETS)
    window_size = int(checkpoint.get("window_size", WINDOW_SIZE))

    missing_features = [column for column in features if column not in data_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in data_df: {missing_features}")

    working_df = data_df.copy()
    if "time" in working_df.columns:
        working_df = working_df.sort_values("time").reset_index(drop=True)
    else:
        working_df = working_df.sort_index().reset_index(drop=True)

    X_raw = working_df[features]
    X_scaled = scaler.transform(X_raw)
    X_tensor = _build_feature_windows(X_scaled, window_size=window_size).to(device)

    with torch.no_grad():
        y_pred = model(X_tensor).detach().cpu().numpy()

    if bool(checkpoint.get("target_scaled", False)):
        target_mean = np.asarray(checkpoint["target_mean"])
        target_std = np.asarray(checkpoint["target_std"])
        y_pred = (y_pred * target_std) + target_mean

    prediction_index = working_df.index[window_size - 1:]
    if "time" in working_df.columns:
        prediction_index = working_df.loc[prediction_index, "time"]

    predictions_df = pd.DataFrame(y_pred, index=prediction_index, columns=targets)
    predictions_df.index.name = "time"
    return predictions_df


def retrain_lstm_from_df(
    data_df: pd.DataFrame,
    validation_fraction: float = 0.2,
    validation_start_date: str | None = None,
    checkpoint_path: str | Path = MODEL_PATH,
    scaler_path: str | Path = SCALER_PATH,
) -> dict:
    """Retrain the LSTM model from scratch using a raw dataframe."""
    return train_lstm_model(
        data_df=data_df,
        validation_fraction=validation_fraction,
        validation_start_date=validation_start_date,
        checkpoint_path=str(checkpoint_path),
        scaler_path=str(scaler_path),
    )


def run_pipeline(
    action: str,
    data_path: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    country_code: str = "DE",
    locations: int = 3,
    api_key: str | None = None,
    validation_fraction: float = 0.2,
    validation_start_date: str | None = None,
    model_path: str | Path = MODEL_PATH,
    scaler_path: str | Path = SCALER_PATH,
):
    """Load data and run either prediction or retraining."""
    data_df = load_data_df(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        country_code=country_code,
        locations=locations,
        api_key=api_key,
    )

    if action == "predict":
        predictions_df = predict_with_lstm(
            data_df=data_df,
            model_path=model_path,
            scaler_path=scaler_path,
        )
        print(predictions_df.tail())
        return predictions_df

    if action == "retrain":
        training_result = retrain_lstm_from_df(
            data_df=data_df,
            validation_fraction=validation_fraction,
            validation_start_date=validation_start_date,
            checkpoint_path=model_path,
            scaler_path=scaler_path,
        )
        print(
            f"Retraining finished. Best epoch: {training_result['best_epoch']}, "
            f"best validation MSE: {training_result['best_val_mse']:.6f}"
        )
        return training_result

    raise ValueError("action must be either 'predict' or 'retrain'")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production pipeline for multiday LSTM prediction or retraining")
    parser.add_argument("action", choices=["predict", "retrain"])
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw input dataframe (.parquet or .csv)")
    parser.add_argument("--start-date", type=str, default=None, help="Fetch start date, e.g. 2018-01-01")
    parser.add_argument("--end-date", type=str, default=None, help="Fetch end date, e.g. 2026-03-13")
    parser.add_argument("--country-code", type=str, default="DE")
    parser.add_argument("--locations", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-start-date", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--scaler-path", type=str, default=str(SCALER_PATH))
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_pipeline(
        action=args.action,
        data_path=args.data_path,
        start_date=args.start_date,
        end_date=args.end_date,
        country_code=args.country_code,
        locations=args.locations,
        validation_fraction=args.validation_fraction,
        validation_start_date=args.validation_start_date,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
    )