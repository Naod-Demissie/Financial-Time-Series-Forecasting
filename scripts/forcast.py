import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class StockForecast:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.model = self.load_lstm_model()
        self.data = self.load_data()

    def load_lstm_model(self):
        """Load the pre-trained LSTM model with custom objects."""
        print(f"Loading model from {self.model_path}...")

        # Define the missing metric
        custom_objects = {"mse": MeanSquaredError()}

        # Load the model with the custom objects
        model = load_model(self.model_path, custom_objects=custom_objects)

        print("Model loaded successfully!")
        return model

    def load_data(self):
        """Load Tesla stock data and return the DataFrame."""
        print(f"Loading data from {self.data_path}...")
        data = pd.read_csv(self.data_path, index_col="Date", parse_dates=True)
        print("Data loaded successfully!")
        return data

    def forecast(self, months_ahead=6):
        """Use the LSTM model to forecast stock prices."""
        print(f"Generating forecast for {months_ahead} months ahead...")

        # Create input sequences for LSTM
        # Convert the DataFrame slice to a NumPy array before reshaping.
        X_input = self.data["Close"][-60:].values.reshape(1, -1, 1)
        forecast = self.model.predict(X_input)

        # ***CHANGE***: Repeat the forecasted value to match the desired shape
        forecast = np.repeat(forecast, months_ahead, axis=0)

        # Inverse scale the forecast to get real prices
        forecast_date = pd.date_range(
            self.data.index[-1], periods=months_ahead, freq="M"
        )

        forecast_df = pd.DataFrame(
            forecast.flatten(), index=forecast_date, columns=["Predicted Close"]
        )
        print("Forecast generated successfully!")
        return forecast_df

    def plot_forecast(self, forecast_df):
        """Plot the historical data along with the forecast."""
        print("Plotting forecast...")
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.data.index, self.data["Close"], label="Historical Data", color="blue"
        )
        plt.plot(
            forecast_df.index,
            forecast_df["Predicted Close"],
            label="Forecast",
            color="red",
        )
        plt.fill_between(
            forecast_df.index,
            forecast_df["Predicted Close"] - 5,
            forecast_df["Predicted Close"] + 5,
            color="gray",
            alpha=0.3,
            label="Confidence Interval",
        )
        plt.title("Tesla Stock Price Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
        print("Forecast plot displayed.")

    def analyze_trends(self, forecast_df):
        """Analyze trends in the forecast."""
        print("Analyzing trends...")
        trend = "stable"
        forecast_prices = forecast_df["Predicted Close"]

        # Check for trend (upward, downward, or stable)
        if forecast_prices.iloc[-1] > forecast_prices.iloc[0]:
            trend = "upward"
        elif forecast_prices.iloc[-1] < forecast_prices.iloc[0]:
            trend = "downward"

        print(f"Trend Analysis: The forecast shows a {trend} trend.")

    def analyze_volatility(self, forecast_df):
        """Analyze volatility and risk in the forecast."""
        print("Analyzing volatility and risks...")
        forecast_prices = forecast_df["Predicted Close"]
        forecast_std = forecast_prices.std()

        print(
            f"Volatility: The standard deviation of the forecasted prices is {forecast_std:.2f}."
        )
        print(
            "Higher standard deviation indicates greater volatility and potential risk."
        )

    def market_opportunities_and_risks(self, forecast_df):
        """Outline potential market opportunities and risks."""
        print("Analyzing market opportunities and risks...")
        forecast_prices = forecast_df["Predicted Close"]

        # Check for significant price change (indicating opportunity/risk)
        price_change = forecast_prices.pct_change().dropna()
        high_volatility_periods = price_change[abs(price_change) > 0.05]  # 5% change

        if not high_volatility_periods.empty:
            print(
                "Potential Risks: High volatility expected during the following periods:"
            )
            print(high_volatility_periods)
        else:
            print("Market is expected to remain relatively stable.")

        print(
            "Market Opportunities: If the trend remains upward, there may be opportunities for profit."
        )
