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

## `train_models.py`  
This script trains classical machine learning models for financial forecasting:  
- **Model Initialization**: Initializes MLflow tracking for experiment logging.  
- **Load Data**: Loads training, validation, and test datasets.  
- **Model Training**: Trains models such as Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.  
- **Evaluation**: Evaluates model performance using metrics like confusion matrix, classification report, and accuracy score.  


## `forcast.py`  
This script uses the trained LSTM model to forecast stock prices:  
- **Load LSTM Model**: Loads the pre-trained LSTM model.  
- **Load Data**: Loads the stock data from CSV files.  
- **Forecast**: Generates forecasts for the specified number of months ahead.  
- **Plot Forecast**: Plots the historical data along with the forecasted values.  
- **Analyze Trends**: Analyzes trends in the forecasted data.  
- **Analyze Volatility**: Analyzes volatility and risk in the forecasted data.  
- **Market Opportunities and Risks**: Outlines potential market opportunities and risks based on the forecasted data.  



## `optimize_portfolio.py`  
This script optimizes a financial portfolio based on the forecasted data:  
- **Load LSTM Model**: Loads the pre-trained LSTM model.  
- **Load Data**: Loads the stock data for TSLA, BND, and SPY from CSV files.  
- **Forecast Prices**: Generates multi-step price forecasts using recursive forecasting.  
- **Compute Returns**: Calculates daily log returns from the forecasted prices.  
- **Compute Annual Return**: Computes annualized return for each asset.  
- **Compute Covariance Matrix**: Computes the covariance matrix of asset returns.  
- **Portfolio Performance**: Calculates portfolio return and volatility given weights.  
- **Optimize Portfolio**: Finds optimal weights to maximize the Sharpe Ratio.  
- **Visualize Portfolio Performance**: Visualizes portfolio performance over time.  
- **Summary**: Summarizes expected return, volatility, Sharpe Ratio, and risk adjustments.