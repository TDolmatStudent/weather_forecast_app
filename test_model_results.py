import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta

from constants import POLYNOMIAL_DEGREE

def main():
    # Load the dataset
    df = pd.read_csv('datasets/cleaned_warsaw_weather_dataset.csv', sep=';')
    
    # Load the polynomial regression model
    model_max = joblib.load('models/model_temp_max.pkl')
    model_min = joblib.load('models/model_temp_min.pkl')
    
    # Generate X values for prediction (all days in the dataset)
    X_all = df[['year', 'month', 'day', 'was_raining_previous_day']]
    X_poly_all = polynomial_features_transform(X_all)
    
    # Predictions for temp_max and temp_min
    y_pred_max = model_max.predict(X_poly_all)
    y_pred_min = model_min.predict(X_poly_all)
    

    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['temp_max'], label='Actual Max Temp', color='blue',)
    plt.plot(df['date'], df['temp_min'], label='Actual Min Temp', color='green')
    
    # Generate dates for the next one year


    start_date = df.iloc[-1]['date'] + timedelta(days=1)
    print(start_date)
    end_date = start_date + timedelta(days=365)
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create DataFrame for future dates with default values for other features
    future_df = pd.DataFrame({'date': future_dates})
    future_df['year'] = future_df['date'].dt.year
    future_df['month'] = future_df['date'].dt.month
    future_df['day'] = future_df['date'].dt.day
    future_df['was_raining_previous_day'] = 0  # Set default value
    
    # Transform features into polynomial features for future dates
    X_poly_future = polynomial_features_transform(future_df[['year', 'month', 'day', 'was_raining_previous_day']])
    
    # Predictions for temp_max and temp_min for future dates
    future_df['temp_max'] = y_pred_max_future = model_max.predict(X_poly_future)
    future_df['temp_min'] = y_pred_min_future = model_min.predict(X_poly_future)
    
    # Plot predicted values for future dates
    plt.plot(future_dates, y_pred_max_future, linestyle='--', color='orange', label='Predicted Max Temp (Future)')
    plt.plot(future_dates, y_pred_min_future, linestyle='--', color='red', label='Predicted Min Temp (Future)')
    # plt.plot(future_dates, future_df['temp_max'].rolling(window=30).mean(), linestyle='--', color='orange', label='Predicted Max Temp (Future)')
    # plt.plot(future_dates, future_df['temp_min'].rolling(window=30).mean(), linestyle='--', color='red', label='Predicted Min Temp (Future)')
    
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title('Actual vs. Predicted Temperatures')
    plt.legend()
    plt.show()

def polynomial_features_transform(X):
    # Transform features into polynomial features
    polynomial_features = PolynomialFeatures(degree=POLYNOMIAL_DEGREE)
    X_poly = polynomial_features.fit_transform(X)
    return X_poly

if __name__ == '__main__':
    main()
