import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def run_data_splitting():
    """
    This function loads the feature engineered data, splits it into
    training and testing sets, and saves the processed data.
    """
    input_data_dir = os.path.join("data", "interim")
    df = pd.read_csv(os.path.join(input_data_dir, "feature_engineered.csv"))

    X = df.drop(["price"], axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    output_data_dir = os.path.join("data", "processed")
    os.makedirs(output_data_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_data_dir, "X_train_processed.csv"), index=False)
    X_test.to_csv(os.path.join(output_data_dir, "X_test_processed.csv"), index=False)
    y_train.to_csv(os.path.join(output_data_dir, "y_train_processed.csv"), index=False)
    y_test.to_csv(os.path.join(output_data_dir, "y_test_processed.csv"), index=False)
    print("Data splitting complete and files saved to data/processed.")

if __name__ == "__main__":
    run_data_splitting()
