import pandas as pd

from constants import CLEANED_DATASET_PATH, ORIGINAL_DATASET_PATH, VISUALIZATION_DATASET_PATH


def clean_for_regression():
    # === Cleaning data for regression model ===

    df = pd.read_csv(ORIGINAL_DATASET_PATH, sep=';')
    df = df.sort_values(by='date', ascending=True)

    df[['year', 'month', 'day']] = df['date'].str.split('-', expand=True)

    df['rain'] = df['rain'].str.replace(',', '.').astype(float)
    df['was_raining_previous_day'] = df['rain'].shift(-1).fillna(0).apply(lambda x: 1 if x > 0.5 else 0)

    df = df.drop(columns=['date'])
    df = df.drop(columns=['wind'])
    df = df.drop(columns=['rain'])

    columns_order = ['year', 'month', 'day'] + [col for col in df.columns if col not in ['year', 'month', 'day']]

    df = df[columns_order]

    df.to_csv(CLEANED_DATASET_PATH, sep=';', index=False)


def clean_for_visualization():
    # === Cleaning data for visualization purposees ===
    df = pd.read_csv(ORIGINAL_DATASET_PATH, sep=';')
    df = df.sort_values(by='date', ascending=True)
    df = df.drop(columns=['wind'])
    df['rain'] = df['rain'].str.replace(',', '.').astype(float)

    df.to_csv(VISUALIZATION_DATASET_PATH, sep=';', index=False)

if __name__ == '__main__':
    clean_for_regression()
    clean_for_visualization()