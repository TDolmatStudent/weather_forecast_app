import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from constants import POLYNOMIAL_DEGREE

def predict_user_input(user_input_data, model_file_max, model_file_min):
    # Load the user input data
    user_input_df = pd.DataFrame(user_input_data, index=[0])
    
    # Load the trained polynomial regression models
    model_max = joblib.load(model_file_max)
    model_min = joblib.load(model_file_min)
    
    # Transform features into polynomial features
    X_poly = polynomial_features_transform(user_input_df)
    
    # Make predictions
    y_pred_max = model_max.predict(X_poly)
    y_pred_min = model_min.predict(X_poly)
    
    # Return the predictions
    return y_pred_max[0], y_pred_min[0]

def polynomial_features_transform(X):
    # Transform features into polynomial features
    polynomial_features = PolynomialFeatures(degree=POLYNOMIAL_DEGREE)
    X_poly = polynomial_features.fit_transform(X)
    return X_poly

if __name__ == '__main__':
    # Example user input data
    user_input_data = {
        'year': 2025,
        'month': 6,
        'day': 1,
        'was_raining_previous_day': 0
    }
    
    # Model files
    model_file_max = 'models/model_temp_max.pkl'
    model_file_min = 'models/model_temp_min.pkl'
    
    # Predict temperature
    pred_max_temp, pred_min_temp = predict_user_input(user_input_data, model_file_max, model_file_min)
    
    print(f'Predicted Max Temperature: {pred_max_temp:.2f} °C')
    print(f'Predicted Min Temperature: {pred_min_temp:.2f} °C')
