# Housing Price Prediction

This project aims to predict housing prices based on various features of the houses. The prediction is done using a Linear Regression model, which has been found to be the most effective for this dataset.

## Dataset

The dataset used in this project is the "Housing Prices Dataset" from Kaggle. It contains information about houses, including their area, number of bedrooms, bathrooms, and other amenities.

You can find the dataset here: [https://www.kaggle.com/datasets/yasserh/housing-prices-dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

## Project Structure

The project is organized as follows:

-   `data/`: Contains the raw, interim, and processed data.
    -   `raw/`: The original, immutable data dump.
    -   `interim/`: Intermediate data that has been transformed.
    -   `processed/`: The final, canonical data sets for modeling.
-   `notebooks/`: Jupyter notebooks for data exploration, feature engineering, data splitting, and model training.
-   `scripts/`: Contains python scripts for feature engineering, data splitting and model training.
-   `requirements.txt`: The requirements file for reproducing the analysis environment.

## Installation and Usage

To run this project, you need to have Python 3 installed. You can set up the environment and run the project by following these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the pipeline:**
    You can run the entire pipeline by executing the notebooks in order, or by running the scripts:
    ```bash
    python scripts/feature_engineering.py
    python scripts/run_data_splitting.py
    python scripts/run_model_training.py
    ```

## Model Performance

Two models were trained and evaluated: Linear Regression and Random Forest. The performance of the models was evaluated using the R-squared metric.

-   **Linear Regression R-squared:** 0.7110
-   **Random Forest R-squared:** 0.6248

The Linear Regression model performed better on this dataset.

## Conclusion

This project demonstrates a complete workflow for a house price prediction task, from data exploration to model training and evaluation. The final model, a Linear Regression, achieves a respectable R-squared score of 0.7110, indicating that it can explain a significant portion of the variance in the housing prices.
