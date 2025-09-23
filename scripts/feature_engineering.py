import os
import pandas as pd
import numpy as np

def feature_engineer(input_data_path, output_data_path):
    """
    This function performs feature engineering on the raw housing data.
    It handles categorical variables, skewness, outliers, and saves the
    processed data.
    """
    # Load the raw data
    df = pd.read_csv(input_data_path)

    # Handling Binary Categorical Features
    binary_columns = [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'
    ]
    for col in binary_columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Handling Multi-Level Categorical Features (with fix for dtype)
    dummies = pd.get_dummies(df['furnishingstatus'], drop_first=True, dtype=int)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop("furnishingstatus", axis=1)

    # Handling Skewness in Numerical Features
    df['price'] = np.log(df['price'])
    df['area'] = np.log(df['area'])

    # Handling Outliers
    q99 = df['area'].quantile(0.99)
    df = df[df['area'] < q99]

    # Saving the Processed Data
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df.to_csv(output_data_path, index=False)
    print(f"Feature engineered data saved to {output_data_path}")

if __name__ == "__main__":
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute paths
    raw_data_path = os.path.join(script_dir, "..", "data", "raw", "Housing.csv")
    engineered_data_path = os.path.join(script_dir, "..", "data", "interim", "feature_engineered.csv")
    feature_engineer(raw_data_path, engineered_data_path)
