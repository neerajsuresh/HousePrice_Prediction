# HousePrice_Prediction
# Project Overview

The project aims to build a machine learning model that predicts housing prices based on a given dataset. This dataset consists of 506 rows and 14 columns, providing the foundation for our analysis.

## Project Objective

This is a regression task, with the goal of predicting the median value of owner-occupied homes (MEDV) in thousands of dollars.

## Libraries Used

- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- plotly
- joblib

## Project Steps

### Data Preprocessing and Exploration

1. Import necessary libraries for data analysis and visualization.
2. Load the dataset and perform basic checks, including checking for missing values, duplicates, and zero variance columns.
3. Explore the distribution of the target variable (MEDV) and other features using visualizations.

### Feature Selection

1. Use SelectKBest with the f_regression scoring function to select the top 10 best features for modeling.
2. Remove less important features from the dataset.

### Data Scaling and Splitting

1. Scale the selected features using MinMaxScaler.
2. Split the dataset into training and testing sets for model training and evaluation.

### Model Building and Evaluation

1. Build and evaluate machine learning models for regression:
   - XGBoost Regressor
   - Extra Trees Regressor
   - Random Forest Regressor
   - LightGBM Regressor
2. Evaluate model performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.

### Model Tuning

1. Perform hyperparameter tuning using techniques like RandomizedSearchCV to improve model performance.

### Model Selection

1. Choose the best-performing model based on evaluation metrics. In this case, the XGBoost Regressor performed well with an R-squared value of 0.8985.

### User Interface (GUI)

1. Create a graphical user interface (GUI) for users to input data and get housing price predictions from the model.

### Model Deployment (Future Work)

1. Explore deploying the chosen model as an API for integration into websites or applications.

## Project Results

The project successfully built a regression model using XGBoost that can predict housing prices based on various features. The model achieved an R-squared value of 0.8985 on the test data, indicating strong predictive performance.

## Future Work

Future work could include deploying the model as an API for easy integration into applications or websites, making it accessible to a broader audience.
