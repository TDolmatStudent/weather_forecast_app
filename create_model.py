import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from constants import POLYNOMIAL_DEGREE


def main():
    df = pd.read_csv('datasets/cleaned_warsaw_weather_dataset.csv', sep=';')

    create_test_and_save_temp_model(df, 'temp_max')
    create_test_and_save_temp_model(df, 'temp_min')


def create_test_and_save_temp_model(df, predict_feature='temp_max'):
    # --- Create ---
    X = df[['year', 'month', 'day', 'was_raining_previous_day']]
    y = df[predict_feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    nomial_features = PolynomialFeatures(degree=POLYNOMIAL_DEGREE)
    X_train_ = nomial_features.fit_transform(X_train)
    X_test_ = nomial_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_, y_train)

    # --- Test ---

    y_pred = model.predict(X_test_)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{predict_feature} Prediction - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}')

    # --- Save ---
    joblib.dump(model, f'models/model_{predict_feature}.pkl')


if __name__ == '__main__':
    main()
