# HousePrice_Prediction
Overview

The project aims to build a machine learning model that predicts housing prices based on a given dataset. This dataset consists of 506 rows and 14 columns, providing the foundation for our analysis.

Project Objective

This is a regression task, with the goal of predicting the median value of owner-occupied homes (MEDV) in thousands of dollars.

Libraries Used

numpy
pandas
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
plotly
joblib

Project Steps

1. Data Preprocessing and Exploration
Import necessary libraries for data analysis and visualization.
Load the dataset and perform basic checks, including checking for missing values, duplicates, and zero variance columns.
Explore the distribution of the target variable (MEDV) and other features using visualizations.
2. Feature Selection
Use SelectKBest with the f_regression scoring function to select the top 10 best features for modeling.
Remove less important features from the dataset.
3. Data Scaling and Splitting
Scale the selected features using MinMaxScaler.
Split the dataset into training and testing sets for model training and evaluation.
4. Model Building and Evaluation
Build and evaluate machine learning models for regression:
XGBoost Regressor
Extra Trees Regressor
Random Forest Regressor
LightGBM Regressor
Evaluate model performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.
5. Model Tuning 
Perform hyperparameter tuning using techniques like RandomizedSearchCV to improve model performance.
6. Model Selection
Choose the best-performing model based on evaluation metrics. In this case, the XGBoost Regressor performed well with an R-squared value of 0.8985.
7. User Interface (GUI)
Create a graphical user interface (GUI) for users to input data and get housing price predictions from the model.
8. Model Deployment (Future Work)
Explore deploying the chosen model as an API for integration into websites or applications.

Project Results

The project successfully built a regression model using XGBoost that can predict housing prices based on various features. The model achieved an R-squared value of 0.8985 on the test data, indicating strong predictive performance.

Future Work

Future work could include deploying the model as an API for easy integration into applications or websites, making it accessible to a broader audience.
