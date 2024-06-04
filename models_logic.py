import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from constants import POLYNOMIAL_DEGREE, MODEL_TEMP_MAX_PATH, MODEL_TEMP_MIN_PATH


def polynomial_features_transform(X):
	# Transforming features into polynomial features
	polynomial_features = PolynomialFeatures(degree=POLYNOMIAL_DEGREE)
	X_poly = polynomial_features.fit_transform(X)
	return X_poly


def predict_temperatures(year, month, day, was_raining_previous_day=0):
    input_data = {
        'year': year,
        'month': month,
        'day': day,
        'was_raining_previous_day': was_raining_previous_day
    }

    input_df = pd.DataFrame(input_data, index=[0])

    model_max = joblib.load(MODEL_TEMP_MAX_PATH)
    model_min = joblib.load(MODEL_TEMP_MIN_PATH)

    X_poly_max = polynomial_features_transform(input_df)
    predicted_temp_max = model_max.predict(X_poly_max)

    input_df['predicted_temp_max'] = predicted_temp_max
    X_poly_min = polynomial_features_transform(input_df)
    predicted_temp_min = model_min.predict(X_poly_min)

    return predicted_temp_max[0], predicted_temp_min[0]


print(predict_temperatures(2024, 6, 11, 0))