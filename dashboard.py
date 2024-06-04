import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime

from models_logic import predict_temperatures

def main():
    st.title('🌡️ Weather prediction app ☀️')

    year = st.number_input('Year:', min_value=2024, max_value=2025)
    month = st.number_input('Month:', min_value=1, max_value=12)
    day = st.number_input('Day:', min_value=1, max_value=31)
    raining_info = st.selectbox('Was it raining yesterday?', ('No information', 'True', 'False'))

    month_str = f'0{month}' if len(str(month)) == 1 else str(month)
    day_str = f'0{day}' if len(str(day)) == 1 else str(day)

    if st.button('Predict Temperature'):
        was_raining_yesterday = 1 if raining_info == 'True' else 0
        predicted_temp_max, predicted_temp_min = predict_temperatures(year, month, day, was_raining_yesterday)

        st.write(f'Predicted Temperature for {year}-{month_str}-{day_str} in Warsaw: **{predicted_temp_max:.2f} / {predicted_temp_min:.2f} °C**')

if __name__ == '__main__':
    main()
