import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from pmdarima import auto_arima
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class TimeSeriesForecaster:
    def __init__(
        self, file_path, model_type="ARIMA", p=1, d=1, q=1, seasonal_order=(1, 1, 1, 12)
    ):
        self.file_path = file_path
        self.model_type = model_type
        self.p, self.d, self.q = p, d, q
        self.seasonal_order = seasonal_order
        self.model = None
        self.data = None
        self.train_data = None
        self.test_data = None
        self.checkpoint_dir = (
            "/content/drive/MyDrive/10 acadamy/W11 Challenge/checkpoints"
        )
        self.log_dir = "/content/drive/MyDrive/10 acadamy/W11 Challenge/logs/"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Initializing {model_type} model with params: p={p}, d={d}, q={q}")

    def load_data(self):
        print(f"Loading data from {self.file_path}...")
        self.data = pd.read_csv(self.file_path, parse_dates=["Date"], index_col="Date")
        print("Data loaded successfully!")
        display(self.data.head())

    def train_test_split(self, train_size=0.7, val_size=0.15):
        train_end = int(len(self.data) * train_size)
        val_end = train_end + int(len(self.data) * val_size)

        self.train_data = self.data.iloc[:train_end]
        self.val_data = self.data.iloc[train_end:val_end]
        self.test_data = self.data.iloc[val_end:]

        print(
            f"Training size: {len(self.train_data)}, Validation size: {len(self.val_data)}, Testing size: {len(self.test_data)}"
        )

    def optimize_parameters(self):
        print("Optimizing ARIMA parameters using auto_arima...")
        auto_model = auto_arima(
            self.train_data["Close"], seasonal=True, trace=True, stepwise=True
        )
        self.p, self.d, self.q = auto_model.order
        self.seasonal_order = auto_model.seasonal_order
        print(
            f"Optimal Parameters Found: p={self.p}, d={self.d}, q={self.q}, seasonal_order={self.seasonal_order}"
        )

    def _train_lstm(self):
        seq_length = 10
        X_train, y_train = self._prepare_lstm_data(self.train_data["Close"], seq_length)

        self.model = Sequential(
            [
                LSTM(
                    50,
                    activation="relu",
                    return_sequences=True,
                    input_shape=(seq_length, 1),
                ),
                LSTM(50, activation="relu"),
                Dense(1),
            ]
        )

        checkpoint_cb = ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_dir, "lstm_best.h5"),
            save_best_only=True,
            monitor="loss",
            mode="min",
        )
        tensorboard_cb = TensorBoard(log_dir=self.log_dir)

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        self.model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=16,
            verbose=1,
            callbacks=[checkpoint_cb, tensorboard_cb],
        )

    def _prepare_lstm_data(self, series, seq_length):
        X, y = [], []
        series = series.values.reshape(-1, 1)
        for i in range(len(series) - seq_length):
            X.append(series[i : i + seq_length])
            y.append(series[i + seq_length])
        return np.array(X), np.array(y)

    def train_model(self):
        if self.model_type == "ARIMA":
            print("Training ARIMA model...")
            self.model = ARIMA(
                self.train_data["Close"], order=(self.p, self.d, self.q)
            ).fit()
        elif self.model_type == "SARIMA":
            print("Training SARIMA model...")
            self.model = SARIMAX(
                self.train_data["Close"],
                order=(self.p, self.d, self.q),
                seasonal_order=self.seasonal_order,
            ).fit()
        elif self.model_type == "LSTM":
            print("Training LSTM model...")
            self._train_lstm()
        else:
            raise ValueError("Invalid model type. Choose 'ARIMA', 'SARIMA', or 'LSTM'.")

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{self.model_type}_model.h5"
        )
        print(f"Saving model checkpoint to {checkpoint_path}")
        self.model.save(checkpoint_path)
        print("Model training complete!")

    def forecast(self, steps=30):
        print(f"Forecasting next {steps} steps...")
        if self.model_type in ["ARIMA", "SARIMA"]:
            forecast_values = self.model.forecast(steps=steps)
        elif self.model_type == "LSTM":
            forecast_values = self._predict_lstm(steps)
        print("Forecasting complete!")
        return forecast_values

    def _predict_lstm(self, steps):
        seq_length = 10
        data_series = self.train_data["Close"].values[-seq_length:].reshape(-1, 1)
        predictions = []
        for _ in range(steps):
            input_data = np.array(data_series[-seq_length:]).reshape(1, seq_length, 1)
            next_step = self.model.predict(input_data)[0, 0]
            predictions.append(next_step)
            data_series = np.append(data_series, [[next_step]], axis=0)
        return predictions

    def evaluate_model(self):
        print("Evaluating model...")
        predictions = self.forecast(len(self.test_data))
        mae = mean_absolute_error(self.test_data["Close"], predictions)
        rmse = mean_squared_error(self.test_data["Close"], predictions)
        mape = mean_absolute_percentage_error(self.test_data["Close"], predictions)
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
        return mae, rmse, mape

    def plot_forecast(self):
        predictions = self.forecast(len(self.test_data))
        plt.figure(figsize=(10, 5))
        plt.plot(self.data.index, self.data["Close"], label="Actual")
        plt.plot(
            self.test_data.index, predictions, label="Forecast", linestyle="dashed"
        )
        plt.legend()
        plt.title(f"{self.model_type} Forecast vs Actual")
        plt.show()
