import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_and_plot_model_sklearn(model, X_test_scaled, y_test, test_df):
    y_pred_test = model.predict(X_test_scaled)

    plot_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred_test
    }, index=test_df.loc[y_test.index, "time"] if "time" in test_df.columns else y_test.index)
    plot_df = plot_df.sort_index()

    pred_std = float(np.std(y_pred_test))
    if pred_std < 1e-6:
        print("Warning: model predictions are nearly constant. The model may be underfit or checkpoint/data may be mismatched.")

    print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")

    plt.figure(figsize=(14, 5))
    plt.plot(plot_df.index, plot_df["actual"], label="Actual", linewidth=1.8)
    plt.plot(plot_df.index, plot_df["predicted"],
             label="Predicted", linewidth=1.5, alpha=0.85)
    plt.title("Test Set: Predicted vs Actual Load")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(6, 6))
    # plt.scatter(plot_df["actual"], plot_df["predicted"], alpha=0.35)
    # axis_min = min(plot_df["actual"].min(), plot_df["predicted"].min())
    # axis_max = max(plot_df["actual"].max(), plot_df["predicted"].max())
    # plt.plot([axis_min, axis_max], [axis_min, axis_max], "r--", linewidth=1.2)
    # plt.title("Predicted vs Actual Scatter (Test)")
    # plt.xlabel("Actual Load (MW)")
    # plt.ylabel("Predicted Load (MW)")
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.show()


def evaluate_and_plot_model_torch(
    model,
    test_loader,
    y_test,
    test_df,
    device,
    window_size=1,
    target_mean=None,
    target_std=None,
    target_names=None,
    target_plot_index=0,
    plot_all_targets=False,
):
    model.eval()
    y_pred_batches = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred_batches.append(model(X_batch).detach().cpu().numpy())

    if not y_pred_batches:
        raise ValueError("Test loader produced no batches.")

    y_pred_values = np.concatenate(y_pred_batches)
    if y_pred_values.ndim == 2 and y_pred_values.shape[1] == 1:
        y_pred_values = y_pred_values[:, 0]

    y_true_aligned = y_test.iloc[window_size - 1:]
    y_true_values = y_true_aligned.values

    is_multi_target = y_pred_values.ndim == 2

    if is_multi_target:
        if hasattr(y_true_aligned, "columns"):
            inferred_names = list(y_true_aligned.columns)
        else:
            inferred_names = [f"target_{i}" for i in range(y_pred_values.shape[1])]

        if target_names is None:
            target_names_resolved = inferred_names
        else:
            target_names_resolved = list(target_names)

        if len(target_names_resolved) != y_pred_values.shape[1]:
            raise ValueError("Length of target_names must match number of model outputs.")
    else:
        target_names_resolved = []

    if target_mean is not None and target_std is not None:
        target_mean_arr = np.asarray(target_mean)
        target_std_arr = np.asarray(target_std)

        if np.any(target_std_arr <= 0):
            raise ValueError("target_std values must be > 0 for inverse scaling.")

        y_pred_values = (y_pred_values * target_std_arr) + target_mean_arr
        y_true_values = (y_true_values * target_std_arr) + target_mean_arr

    if not is_multi_target:
        y_pred_test = pd.Series(data=y_pred_values, index=y_true_aligned.index)

        plot_df = pd.DataFrame(
            {
                "actual": y_true_values,
                "predicted": y_pred_test.values,
            },
            index=test_df.loc[y_true_aligned.index,
                              "time"] if "time" in test_df.columns else y_true_aligned.index,
        ).sort_index()

        pred_std = float(plot_df["predicted"].std())
        if pred_std < 1e-6:
            print("Warning: model predictions are nearly constant. The loaded checkpoint appears collapsed to a mean prediction.")

        print(
            f"Test MAE: {mean_absolute_error(plot_df['actual'], plot_df['predicted']):.2f}")
        print(
            f"Test RMSE: {np.sqrt(mean_squared_error(plot_df['actual'], plot_df['predicted'])):.2f}")

        plt.figure(figsize=(14, 5))
        plt.plot(plot_df.index, plot_df["actual"], label="Actual", linewidth=1.8)
        plt.plot(plot_df.index, plot_df["predicted"],
                 label="Predicted", linewidth=1.5, alpha=0.85)
        plt.title("Test Set: Predicted vs Actual Load")
        plt.xlabel("Date")
        plt.ylabel("Load (MW)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return plot_df

    y_pred_test = pd.DataFrame(data=y_pred_values, index=y_true_aligned.index, columns=target_names_resolved)
    y_true_df = pd.DataFrame(data=y_true_values, index=y_true_aligned.index, columns=target_names_resolved)

    eval_index = test_df.loc[y_true_aligned.index, "time"] if "time" in test_df.columns else y_true_aligned.index
    y_pred_test.index = eval_index
    y_true_df.index = eval_index

    print("Multi-target metrics:")
    for target_name in target_names_resolved:
        target_mae = mean_absolute_error(y_true_df[target_name], y_pred_test[target_name])
        target_rmse = np.sqrt(mean_squared_error(y_true_df[target_name], y_pred_test[target_name]))
        print(f"{target_name} -> MAE: {target_mae:.2f}, RMSE: {target_rmse:.2f}")

    if plot_all_targets:
        fig, axes = plt.subplots(len(target_names_resolved), 1, figsize=(14, 4 * len(target_names_resolved)), sharex=True)
        if len(target_names_resolved) == 1:
            axes = [axes]

        for idx, target_name in enumerate(target_names_resolved):
            pred_std = float(y_pred_test[target_name].std())
            if pred_std < 1e-6:
                print(f"Warning: model predictions for {target_name} are nearly constant.")

            axes[idx].plot(y_true_df.index, y_true_df[target_name], label=f"Actual ({target_name})", linewidth=1.8)
            axes[idx].plot(
                y_pred_test.index,
                y_pred_test[target_name],
                label=f"Predicted ({target_name})",
                linewidth=1.5,
                alpha=0.85,
            )
            axes[idx].set_title(f"Test Set: Predicted vs Actual Load ({target_name})")
            axes[idx].set_ylabel("Load (MW)")
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

        axes[-1].set_xlabel("Date")
        fig.tight_layout()
        plt.show()
    else:
        if target_plot_index < 0 or target_plot_index >= len(target_names_resolved):
            raise ValueError("target_plot_index is out of range for target_names")

        plot_target = target_names_resolved[target_plot_index]
        pred_std = float(y_pred_test[plot_target].std())
        if pred_std < 1e-6:
            print(f"Warning: model predictions for {plot_target} are nearly constant.")

        plt.figure(figsize=(14, 5))
        plt.plot(y_true_df.index, y_true_df[plot_target], label=f"Actual ({plot_target})", linewidth=1.8)
        plt.plot(y_pred_test.index, y_pred_test[plot_target],
                 label=f"Predicted ({plot_target})", linewidth=1.5, alpha=0.85)
        plt.title(f"Test Set: Predicted vs Actual Load ({plot_target})")
        plt.xlabel("Date")
        plt.ylabel("Load (MW)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    result_df = pd.DataFrame(index=eval_index)
    for target_name in target_names_resolved:
        result_df[f"actual_{target_name}"] = y_true_df[target_name].values
        result_df[f"predicted_{target_name}"] = y_pred_test[target_name].values

    return result_df
