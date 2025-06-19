import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

def run_predictions():
    # === Paths ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "model_input.csv")
    plot_dir = os.path.join(script_dir, "public", "images")
    os.makedirs(plot_dir, exist_ok=True)

    # === Load Data ===
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df[["datetime", "power"]].dropna()
    df.rename(columns={"datetime": "timestamp"}, inplace=True)

    # === Remove Outliers ===
    Q1 = df["power"].quantile(0.25)
    Q3 = df["power"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["power"] >= Q1 - 1.5 * IQR) & (df["power"] <= Q3 + 1.5 * IQR)]

    # === Feature Engineering ===
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    for lag in range(1, 4):
        df[f"power_lag{lag}"] = df["power"].shift(lag)
    df.dropna(inplace=True)

    features = ["hour", "minute", "dayofweek", "month", "power_lag1", "power_lag2", "power_lag3"]
    X = df[features]
    y = df["power"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Evaluation Helper ===
    results = []

    def evaluate_model(name, y_true, y_pred):
        return {
            "name": name,
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            "y_true": y_true,
            "y_pred": y_pred
        }

    # === Train Traditional Models ===
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(),
        "SVR": SVR()
    }
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results.append(evaluate_model(name, y_test, y_pred))

    # === LSTM Model ===
    seq_len = 10
    data = df[["power"]].values
    lstm_scaler = MinMaxScaler()
    data_scaled = lstm_scaler.fit_transform(data)
    split_idx = int(len(data_scaled) * 0.8)
    train_data = data_scaled[:split_idx]
    test_data = data_scaled[split_idx - seq_len:]

    train_gen = TimeseriesGenerator(train_data, train_data, length=seq_len, batch_size=16)
    test_gen = TimeseriesGenerator(test_data, test_data, length=seq_len, batch_size=16)

    lstm_model = Sequential([
        LSTM(64, activation="tanh", return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(train_gen, epochs=50, callbacks=[EarlyStopping(patience=5)], verbose=0)

    preds = lstm_model.predict(test_gen)
    y_true_lstm = test_data[seq_len:]
    results.append(evaluate_model(
        "LSTM",
        pd.Series(lstm_scaler.inverse_transform(y_true_lstm).flatten()),
        pd.Series(lstm_scaler.inverse_transform(preds).flatten())
    ))

    # === Plotting ===
    sns.set_theme(style="whitegrid")
    colors = {"actual": "#4c72b0", "pred": "#55a868", "error": "#c44e52"}

    # 1. Actual vs Predicted
    plt.figure(figsize=(18, 8))
    for i, res in enumerate(results):
        plt.subplot(2, 3, i + 1)
        plt.plot(res["y_true"].values, label="Actual", color=colors["actual"])
        plt.plot(res["y_pred"], label="Predicted", color=colors["pred"], linestyle="--")
        plt.title(res["name"])
        plt.xlabel("Time Step")
        plt.ylabel("Power")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "actual_vs_predicted.png"))

    # 2. Residuals
    plt.figure(figsize=(18, 8))
    for i, res in enumerate(results):
        plt.subplot(2, 3, i + 1)
        residuals = res["y_true"] - res["y_pred"]
        plt.plot(residuals, color=colors["error"])
        plt.title(res["name"])
        plt.xlabel("Time Step")
        plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "residuals.png"))

    # 3. Histogram of Errors
    plt.figure(figsize=(18, 8))
    for i, res in enumerate(results):
        plt.subplot(2, 3, i + 1)
        sns.histplot(res["y_true"] - res["y_pred"], bins=50, kde=True, color=colors["error"])
        plt.title(res["name"])
        plt.xlabel("Prediction Error")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "error_distribution.png"))

    # 4. Scatter
    plt.figure(figsize=(18, 8))
    for i, res in enumerate(results):
        plt.subplot(2, 3, i + 1)
        plt.scatter(res["y_true"], res["y_pred"], alpha=0.5, color=colors["pred"])
        plt.plot([res["y_true"].min(), res["y_true"].max()],
                 [res["y_true"].min(), res["y_true"].max()], "k--")
        plt.title(res["name"])
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "prediction_vs_actual.png"))
