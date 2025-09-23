import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def run_model_training():
    """
    This function loads the processed data, trains and evaluates
    Linear Regression and Random Forest models, and prints the
    R-squared scores.
    """
    # Load the processed data
    input_data_dir = os.path.join("data", "processed")
    X_train = pd.read_csv(os.path.join(input_data_dir, "X_train_processed.csv"))
    y_train = pd.read_csv(os.path.join(input_data_dir, "y_train_processed.csv"))
    X_test = pd.read_csv(os.path.join(input_data_dir, "X_test_processed.csv"))
    y_test = pd.read_csv(os.path.join(input_data_dir, "y_test_processed.csv"))

    # Convert y_train and y_test to 1D arrays
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Train and evaluate the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)

    # Train and evaluate the Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)

    print("--- Final Model Performance Comparison ---")
    print(f"Linear Regression R-squared: {r2_lr:.4f}")
    print(f"Random Forest R-squared: {r2_rf:.4f}")

if __name__ == "__main__":
    run_model_training()
