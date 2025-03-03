# Financial Time Series Forecasting  

This project focuses on forecasting financial time series data using various machine learning models. The project is divided into multiple Jupyter notebooks, each handling different stages of the data processing and modeling pipeline.  

## Notebooks Overview  

### `1.0-Data-Preprocessing.ipynb`  
This notebook handles the initial data preprocessing steps, including:  
- **Importing Libraries**: Loading necessary Python libraries for data manipulation and visualization.  
- **Data Loading**: Reading raw data files into Pandas DataFrames.  
- **Data Inspection**: Inspecting the datasets for general information, uniqueness, missing values, and duplication.  
- **Data Preprocessing**: Handling duplicates, converting data types, handling outliers, and mapping IP addresses to countries.  

### `2.0-Data-Exploration.ipynb`  
This notebook focuses on exploratory data analysis (EDA) to understand the data better:  
- **Import Libraries**: Loading necessary libraries for EDA.  
- **Data Loading**: Loading the processed data from the previous notebook.  
- **Exploratory Data Analysis**: Performing univariate and bivariate analysis to explore relationships between features and the target variable. This includes visualizations like histograms, box plots, and correlation matrices.  

### `3.0-Model-Training.ipynb`  
This notebook covers the training of various machine learning models:  
- **Import Libraries**: Loading necessary libraries for model training.  
- **Data Loading**: Loading the processed data.  
- **Modeling**: Initializing and training classical machine learning models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) and deep learning models (CNN, RNN, LSTM) on the processed data. Evaluating the performance of each model using appropriate metrics.  

### `4.0-Model-Forecast.ipynb`  
This notebook focuses on forecasting future values using the trained models:  
- **Import Libraries**: Loading necessary libraries for forecasting.  
- **Model Forecast**: Using the trained models to forecast future values for the financial time series data. This includes generating forecasts, plotting the forecasted values, and analyzing trends and volatility.
