import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_and_plot_model_sklearn(model, X_test_scaled, y_test, test_df):
    y_pred_test = model.predict(X_test_scaled)

    plot_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred_test
    }, index=test_df.loc[y_test.index, "time"] if "time" in test_df.columns else y_test.index)
    plot_df = plot_df.sort_index()

    print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
    print(f"Test RMSE: {mean_squared_error(y_test, y_pred_test):.2f}")

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