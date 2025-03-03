import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, model_path, data_paths, forecast_horizon=10):
        """
        Initialize the Portfolio Optimizer with the LSTM model and datasets.
        :param model_path: Path to the saved LSTM model.
        :param data_paths: Dictionary containing paths to TSLA, BND, and SPY data.
        :param forecast_horizon: Number of days to forecast.
        """
        self.model_path = model_path
        self.model = self.load_lstm_model()
        self.data = self.load_data(data_paths)
        self.forecast_horizon = forecast_horizon
        self.forecasted_data = self.forecast_prices()
        self.returns = self.compute_returns()
        print("Initialized PortfolioOptimizer with forecasted data.")

    def load_lstm_model(self):
        """Load the pre-trained LSTM model with custom objects."""
        print(f"Loading model from {self.model_path}...")
        custom_objects = {"mse": MeanSquaredError()}
        model = load_model(self.model_path, custom_objects=custom_objects)
        print("Model loaded successfully!")
        return model

    def load_data(self, data_paths):
        """Load datasets for TSLA, BND, and SPY."""
        data = {
            symbol: pd.read_csv(path, index_col="Date", parse_dates=True)
            for symbol, path in data_paths.items()
        }
        print("Loaded datasets for TSLA, BND, and SPY.")
        return data

    def forecast_prices(self):
        """
        Generate multi-step price forecasts using recursive forecasting.
        This method creates a time-series of forecasted prices for each asset.
        """
        forecasted_prices = {symbol: [] for symbol in self.data.keys()}

        # For each asset, use the last 60 days of prices as the initial input sequence.
        for symbol, df in self.data.items():
            prices = df["Close"].values.reshape(-1, 1)
            if len(prices) < 60:
                raise ValueError(
                    f"Not enough data for {symbol} to forecast (needs at least 60 days)."
                )

            # Starting input: last 60 days of observed prices
            input_seq = prices[-60:].copy()

            # Recursive forecasting for the defined horizon
            for _ in range(self.forecast_horizon):
                # Predict next price
                pred = self.model.predict(input_seq.reshape(1, 60, 1)).flatten()[0]
                forecasted_prices[symbol].append(pred)
                # Update the sequence: remove the oldest and append the new prediction
                input_seq = np.append(input_seq[1:], [[pred]], axis=0)

        forecast_df = pd.DataFrame(forecasted_prices)
        print("Generated forecasted prices:")
        display(forecast_df.head())
        return forecast_df

    def compute_returns(self):
        """
        Calculate daily log returns from forecasted prices.
        Note: This assumes that the forecasted prices are strictly positive.
        """
        # Compute log returns for each asset and drop the first NaN row
        returns = np.log(self.forecasted_data / self.forecasted_data.shift(1)).dropna()
        print("Computed daily log returns:")
        display(returns.head())
        return returns

    def compute_annual_return(self):
        """Compute annualized return for each asset."""
        avg_daily_return = self.returns.mean()
        annual_return = avg_daily_return * 252
        print("Computed annualized returns:")
        print(annual_return)
        return annual_return

    def compute_covariance_matrix(self):
        """Compute the covariance matrix of asset returns."""
        cov_matrix = self.returns.cov() * 252
        print("Computed annualized covariance matrix:")
        print(cov_matrix)
        return cov_matrix

    def portfolio_performance(self, weights):
        """Calculate portfolio return and volatility given weights."""
        annual_return = np.dot(weights, self.compute_annual_return())
        volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.compute_covariance_matrix(), weights))
        )
        sharpe_ratio = annual_return / volatility
        return -sharpe_ratio  # Negative because we minimize in optimization

    def optimize_portfolio(self):
        """Find optimal weights to maximize the Sharpe Ratio."""
        num_assets = len(self.forecasted_data.columns)
        init_guess = np.ones(num_assets) / num_assets  # Equal weight start
        bounds = [(0, 1) for _ in range(num_assets)]  # No short-selling
        constraints = {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
        }  # Weights must sum to 1

        result = minimize(
            self.portfolio_performance,
            init_guess,
            bounds=bounds,
            constraints=constraints,
        )
        optimal_weights = result.x
        print("Optimized portfolio weights:")
        print(optimal_weights)
        return optimal_weights

    def visualize_portfolio_performance(self, optimal_weights):
        """Visualize portfolio performance over time."""
        portfolio_returns = self.returns.dot(optimal_weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_returns, label="Optimized Portfolio")
        plt.title("Cumulative Portfolio Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.show()
        print("Displayed cumulative return chart.")

    def summary(self, optimal_weights):
        """Summarize expected return, volatility, Sharpe Ratio, and risk adjustments."""
        annual_return = np.dot(optimal_weights, self.compute_annual_return())
        volatility = np.sqrt(
            np.dot(
                optimal_weights.T,
                np.dot(self.compute_covariance_matrix(), optimal_weights),
            )
        )
        sharpe_ratio = annual_return / volatility

        print("Portfolio Optimization Summary:")
        print(f"Expected Annual Return: {annual_return:.4f}")
        print(f"Expected Volatility: {volatility:.4f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
