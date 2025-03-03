import yfinance as yf
import pandas as pd
import numpy as np
from IPython.display import display


class FinancialDataProcessor:
    def __init__(self, tickers, start_date, end_date):
        """
        Initialize the data processor with asset tickers and a date range.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        print(
            f"Initialized processor for {', '.join(self.tickers)} from {self.start_date} to {self.end_date}."
        )

    def fetch_data(self):
        """
        Fetch historical data for the given assets from Yahoo Finance.
        """
        print("Fetching data from Yahoo Finance...")
        for ticker in self.tickers:
            print(f"Downloading data for {ticker}...")
            self.data[ticker] = yf.download(
                ticker, start=self.start_date, end=self.end_date
            )
        print("Data fetching complete.")

    def basic_statistics(self):
        """
        Display basic statistics and summary of the data.
        """
        print("Generating basic statistics...")
        for ticker, df in self.data.items():
            print(f"\nStatistics for {ticker}:")
            display(df.describe())

    def check_missing_values(self):
        """
        Check for missing values in the dataset.
        """
        print("Checking for missing values...")
        for ticker, df in self.data.items():
            missing = df.isnull().sum()
            if missing.any():
                print(f"{ticker} has missing values:\n{missing}")
            else:
                print(f"{ticker} has no missing values.")

    def handle_missing_values(self, method="interpolate"):
        """
        Handle missing values using a specified method.
        """
        print(f"Handling missing values using {method} method...")
        for ticker, df in self.data.items():
            if method == "drop":
                self.data[ticker] = df.dropna()
            elif method == "fill":
                self.data[ticker] = df.fillna(method="bfill").fillna(method="ffill")
            elif method == "interpolate":
                self.data[ticker] = df.interpolate()
            print(f"Missing values handled for {ticker}.")

    def normalize_data(self):
        """
        Normalize the dataset using Min-Max scaling.
        """
        print("Normalizing data using Min-Max scaling...")
        for ticker, df in self.data.items():
            self.data[ticker] = (df - df.min()) / (df.max() - df.min())
        print("Data normalization complete.")

    def save_cleaned_data(self, base_filename):
        """
        Save the cleaned and processed data to separate CSV files for each ticker.
        """
        print("Saving data to files...")
        for ticker, df in self.data.items():
            filename = f"{base_filename}{ticker}.csv"
            df.to_csv(filename)
            print(f"Data for {ticker} saved to {filename}.")
