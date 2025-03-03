# Financial-Time-Series-Forecasting

This project focuses on time series forecasting for financial assets, including Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY), using historical data from YFinance. It involves preprocessing data, developing predictive models (ARIMA, SARIMA, or LSTM), and optimizing portfolio allocation to maximize returns while managing risk.

## Project Structure


```
├── data
│   ├── processed
│   └── raw
├── logs
├── notebooks
│   ├── 1.0-data-preprocessing.ipynb
│   ├── 2.0-data-exploration.ipynb
│   ├── 3.0-model-training.ipynb
│   ├── 4.0-model-forcast.ipynb
│   └── README.md
├── scripts
│   ├── __init__.py
│   ├── explore_data.py
│   ├── preprocess_data.py
│   ├── train_models.py
│   ├── forcast.py
│   └── README.md
├── src
│   ├── __init__.py
│   └── README.md
├── tests
│  └── __init__.py
├── checkpoints
├── README.md
├── requirements.txt
```


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Naod-Demissie/Financial-Time-Series-Forecasting.git
   cd Financial-Time-Series-Forecasting
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv/Scripts/activate`
   pip install -r requirements.txt
   ```
