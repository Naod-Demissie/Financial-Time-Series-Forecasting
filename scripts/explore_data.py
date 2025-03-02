import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm


class FinancialEDA:
    def __init__(self, tsla_path, bnd_path, spy_path):
        self.tsla_data = pd.read_csv(tsla_path, parse_dates=["Date"], index_col="Date")
        self.bnd_data = pd.read_csv(bnd_path, parse_dates=["Date"], index_col="Date")
        self.spy_data = pd.read_csv(spy_path, parse_dates=["Date"], index_col="Date")
        self.assets = {
            "TSLA": self.tsla_data,
            "BND": self.bnd_data,
            "SPY": self.spy_data,
        }

    def plot_closing_prices(self):
        plt.figure(figsize=(12, 6))
        for asset, data in self.assets.items():
            plt.plot(data.index, data["Close"], label=asset)
        plt.title("Closing Prices Over Time")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend()
        plt.show()

    def calculate_daily_returns(self):
        for asset, data in self.assets.items():
            data["Daily Return"] = data["Close"].pct_change()

    def plot_daily_returns(self):
        plt.figure(figsize=(12, 6))
        for asset, data in self.assets.items():
            plt.plot(data.index, data["Daily Return"], label=asset)
        plt.axhline(y=0, color="black", linestyle="--")
        plt.title("Daily Percentage Change")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.legend()
        plt.show()

    def rolling_statistics(self, window=30):
        for asset, data in self.assets.items():
            data[f"Rolling Mean ({window} days)"] = (
                data["Close"].rolling(window=window).mean()
            )
            data[f"Rolling Std ({window} days)"] = (
                data["Close"].rolling(window=window).std()
            )

            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data["Close"], label="Closing Price")
            plt.plot(
                data.index,
                data[f"Rolling Mean ({window} days)"],
                label="Rolling Mean",
                linestyle="--",
            )
            plt.fill_between(
                data.index,
                data[f"Rolling Mean ({window} days)"]
                - data[f"Rolling Std ({window} days)"],
                data[f"Rolling Mean ({window} days)"]
                + data[f"Rolling Std ({window} days)"],
                color="gray",
                alpha=0.3,
                label="Rolling Std Dev",
            )
            plt.title(f"Rolling Mean & Std Dev for {asset}")
            plt.legend()
            plt.show()

    def detect_outliers(self):
        for asset, data in self.assets.items():
            mean, std = data["Daily Return"].mean(), data["Daily Return"].std()
            data["Outlier"] = np.abs(data["Daily Return"] - mean) > (2 * std)
            outliers = data[data["Outlier"]]

            plt.figure(figsize=(12, 6))
            plt.plot(
                data.index, data["Daily Return"], label="Daily Return", color="blue"
            )
            plt.scatter(
                outliers.index,
                outliers["Daily Return"],
                color="red",
                label="Outliers",
                zorder=3,
            )
            plt.axhline(y=mean, color="green", linestyle="--", label="Mean Return")
            plt.axhline(
                y=mean + 2 * std, color="red", linestyle="--", label="Upper Bound"
            )
            plt.axhline(
                y=mean - 2 * std, color="red", linestyle="--", label="Lower Bound"
            )
            plt.title(f"Outlier Detection for {asset}")
            plt.legend()
            plt.show()

    def seasonal_decompose(self):
        for asset, data in self.assets.items():
            decomposition = sm.tsa.seasonal_decompose(
                data["Close"].dropna(), model="additive", period=252
            )
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            decomposition.observed.plot(ax=axes[0], title=f"{asset} - Observed")
            decomposition.trend.plot(ax=axes[1], title=f"{asset} - Trend")
            decomposition.seasonal.plot(ax=axes[2], title=f"{asset} - Seasonal")
            decomposition.resid.plot(ax=axes[3], title=f"{asset} - Residual")
            plt.tight_layout()
            plt.show()

    def risk_analysis(self):
        risk_free_rate = 0.02 / 252  # Daily risk-free rate (assuming 2% annualized)

        for asset, data in self.assets.items():
            mean_return, std_dev = (
                data["Daily Return"].mean(),
                data["Daily Return"].std(),
            )
            var_95 = norm.ppf(0.05, mean_return, std_dev)
            sharpe_ratio = (mean_return - risk_free_rate) / std_dev

            print(f"Risk Metrics for {asset}:")
            print(f"Mean Daily Return: {mean_return:.5f}")
            print(f"Standard Deviation: {std_dev:.5f}")
            print(f"95% Value at Risk (VaR): {var_95:.5f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print("-" * 50)
