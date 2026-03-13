try:
    from utils.lstm_model import LSTMForecaster
    from utils.data_preparation import create_torch_dataset
except ImportError:
    from lstm_model import LSTMForecaster
    from data_preparation import create_torch_dataset

import os
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Hyperparameters and device configuration
BATCH_SIZE = 32
WINDOW_SIZE = 21
EPOCHS = 3000
LEARNING_RATE = 5e-4
VALIDATE_EVERY = 10
EARLY_STOPPING_PATIENCE = 20
MIN_IMPROVEMENT = 1e-4

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

FEATURES_TODAY = ['Temp', 'Min Temp', 'Max Temp', 'load']
FEATURES_TARGET_TIME = ['is_holiday', 'dow_sin',
                        'dow_cos', 'month_sin', 'month_cos']

FEATURES = FEATURES_TODAY + FEATURES_TARGET_TIME

TARGETS = ["load_t+1", "load_t+2", "load_t+3", "load_t+4"]
BEST_MODEL_PATH = "models/best_lstm_multiday_load_forecaster.pt"


def train_lstm_model(data_df,
                     window_size=WINDOW_SIZE, 
                     features=FEATURES, targets=TARGETS, 
                     batch_size=BATCH_SIZE, device=DEVICE,
                     lr=LEARNING_RATE, epochs=EPOCHS,
                     min_improvement=MIN_IMPROVEMENT,
                     validate_every=VALIDATE_EVERY,
                     early_stopping_patience=EARLY_STOPPING_PATIENCE,
                     checkpoint_path=BEST_MODEL_PATH,
                     scaler_path="models/best_lstm_multiday_feature_scaler.joblib",
                     validation_fraction=0.2,
                     validation_start_date=None,
                     scale_features=FEATURES_TODAY):
    """Train LSTM forecaster from an unscaled raw dataframe.

    The function sorts data chronologically, splits it into train and validation data,
    scales selected feature columns using only the training split, and trains a fresh model.

    Validation is always used for learning-rate scheduling, model selection, and early stopping.
    Training always starts from scratch; checkpoints are overwritten with the current best run.
    """

    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("data_df must be a pandas DataFrame")

    missing_features = [column for column in features if column not in data_df.columns]
    missing_targets = [column for column in targets if column not in data_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in data_df: {missing_features}")
    if missing_targets:
        raise ValueError(f"Missing target columns in data_df: {missing_targets}")

    working_df = data_df.copy()
    if "time" in working_df.columns:
        working_df = working_df.sort_values("time").reset_index(drop=True)
    else:
        working_df = working_df.sort_index().reset_index(drop=True)

    if validation_start_date is not None:
        if "time" not in working_df.columns:
            raise ValueError("validation_start_date requires a 'time' column in data_df")
        validation_start_date = pd.Timestamp(validation_start_date)
        train_df = working_df[working_df["time"] < validation_start_date].copy()
        val_df = working_df[working_df["time"] >= validation_start_date].copy()
    else:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        split_idx = int(len(working_df) * (1 - validation_fraction))
        train_df = working_df.iloc[:split_idx].copy()
        val_df = working_df.iloc[split_idx:].copy()

    if len(train_df) <= window_size:
        raise ValueError("Training split is too small for the configured window_size")
    if len(val_df) <= window_size:
        raise ValueError("Validation split is too small for the configured window_size")

    X_train = train_df[features]
    X_val = val_df[features]
    y_train = train_df[targets].copy()
    y_val = val_df[targets].copy()

    scaler = ColumnTransformer(
        transformers=[("num", StandardScaler(), scale_features)],
        remainder="passthrough",
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    model = LSTMForecaster(
        input_size=len(features),
        hidden_size=64,
        num_layers=1,
        output_size=len(targets),
        dropout=0.0,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        threshold=min_improvement,
        threshold_mode="abs",
    )

    # Scale each target independently using train split statistics
    target_mean = y_train.mean(axis=0)
    target_std = y_train.std(axis=0)

    if (target_std <= 0).any():
        raise ValueError("All target standard deviations must be > 0 for scaling.")

    y_train_scaled = (y_train - target_mean) / target_std
    y_val_scaled = (y_val - target_mean) / target_std

    print("Target scaling ->")
    print("mean:", target_mean.round(2).to_dict())
    print("std:", target_std.round(2).to_dict())

    # Training Data
    X_train_tensor, y_train_tensor = create_torch_dataset(
        X=X_train_scaled, y=y_train_scaled, window_size=window_size)
    training_loader = data.DataLoader(
        data.TensorDataset(X_train_tensor, y_train_tensor),
        shuffle=True,
        batch_size=batch_size,
    )

    # Validation Data
    X_val_tensor, y_val_tensor = create_torch_dataset(
        X=X_val_scaled, y=y_val_scaled, window_size=window_size)
    validation_loader = data.DataLoader(
        data.TensorDataset(X_val_tensor, y_val_tensor),
        shuffle=False,
        batch_size=batch_size,
    )

    best_val_mse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_mse_sum = 0.0
        train_mae_sum = 0.0
        train_elements = 0

        for X_batch, y_batch in training_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_elements = y_batch.numel()
            train_elements += batch_elements
            train_mse_sum += torch.sum((y_pred - y_batch) ** 2).item()
            train_mae_sum += torch.sum(torch.abs(y_pred - y_batch)).item()

        train_mse = train_mse_sum / train_elements
        train_rmse = np.sqrt(train_mse)
        train_mae = train_mae_sum / train_elements

        if epoch % validate_every != 0:
            continue

        model.eval()
        val_mse_sum = 0.0
        val_mae_sum = 0.0
        val_elements = 0
        val_pred_sum = 0.0
        val_pred_sq_sum = 0.0

        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)

                batch_elements = y_batch.numel()
                val_elements += batch_elements
                val_mse_sum += torch.sum((y_pred - y_batch) ** 2).item()
                val_mae_sum += torch.sum(torch.abs(y_pred - y_batch)).item()
                val_pred_sum += torch.sum(y_pred).item()
                val_pred_sq_sum += torch.sum(y_pred ** 2).item()

        val_mse = val_mse_sum / val_elements
        val_rmse = np.sqrt(val_mse)
        val_mae = val_mae_sum / val_elements

        val_pred_mean = val_pred_sum / val_elements
        val_pred_var = max((val_pred_sq_sum / val_elements) - (val_pred_mean ** 2), 0.0)
        val_pred_std = np.sqrt(val_pred_var)

        scheduler.step(val_mse)

        improved = val_mse < (best_val_mse - min_improvement)
        if improved:
            best_val_mse = val_mse
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_scaled": True,
                    "target_mean": target_mean.to_numpy().tolist(),
                    "target_std": target_std.to_numpy().tolist(),
                    "window_size": window_size,
                    "features": features,
                    "targets": targets,
                },
                checkpoint_path,
            )
            epochs_without_improvement = 0
            checkpoint_note = " [saved best]"
        else:
            epochs_without_improvement += 1
            checkpoint_note = ""

        print(
            f"Epoch {epoch}: train RMSE {train_rmse:.4f}, train MAE {train_mae:.4f}, "
            f"val RMSE {val_rmse:.4f}, val MAE {val_mae:.4f}, val MSE {val_mse:.4f}, "
            f"val pred std {val_pred_std:.2f}, lr {optimizer.param_groups[0]['lr']:.2e}{checkpoint_note}"
        )

        if val_pred_std < 0.05:
            print("Warning: validation prediction std is very low; model may be collapsing to near-constant output.")

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. No sufficient improvement for "
                f"{early_stopping_patience} validation checks."
            )
            break

    print(f"Best model saved to {checkpoint_path} (epoch {best_epoch}, val MSE {best_val_mse:.6f})")

    return {
        "model": model,
        "scaler": scaler,
        "train_df": train_df,
        "val_df": val_df,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "checkpoint_path": checkpoint_path,
        "scaler_path": scaler_path,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "target_mean": target_mean,
        "target_std": target_std,
        "resumed": False,
    }
