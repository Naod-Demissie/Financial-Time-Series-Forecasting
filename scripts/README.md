# Scripts Overview  

This directory contains Python scripts for preprocessing financial data, exploring data, and training machine learning models. Below is a brief description of each script:  

## `preprocess_data.py`  
This script handles the initial data preprocessing steps:  
- **Fetching Data**: Downloads historical data for specified assets from Yahoo Finance.  
- **Basic Statistics**: Displays basic statistics and summaries of the data.  
- **Missing Values**: Checks for and handles missing values using various methods.  
- **Normalization**: Normalizes the data using Min-Max scaling.  
- **Saving Data**: Saves the cleaned and processed data to CSV files.  

## `explore_data.py`  
This script performs exploratory data analysis (EDA) on financial data:  
- **Plot Closing Prices**: Visualizes the closing prices of assets over time.  
- **Calculate Daily Returns**: Computes daily percentage changes in asset prices.  
- **Plot Daily Returns**: Visualizes daily returns.  
- **Rolling Statistics**: Calculates and plots rolling mean and standard deviation.  
- **Outlier Detection**: Identifies and visualizes outliers in daily returns.  
- **Seasonal Decomposition**: Decomposes time series data into trend, seasonal, and residual components.  
- **Risk Analysis**: Computes risk metrics such as mean return, standard deviation, Value at Risk (VaR), and Sharpe ratio.  
