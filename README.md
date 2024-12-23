# Loan Repayment Prediction Model

This project aims to predict whether a borrower will repay their loan within 5 days based on various features. The project uses classification algorithms to build a model and predict the likelihood of loan repayment.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Prerequisites](#Prerequisites)
- [Conclusion](#conclusion)

## Project Overview

The goal of this project is to develop a model that predicts the likelihood of loan repayment within 5 days, using a dataset from a microfinance institution collaborating with a telecom company. The project performs data cleaning, feature engineering, model selection, evaluation, and hyperparameter tuning to ensure the best possible predictions.

## Technologies Used

- **Python** (v3.8+)
- **PyCaret** for automating model selection and evaluation
- **Scikit-Learn** for machine learning models and metrics
- **XGBoost**, **CatBoost**, **LightGBM**, **ExtraTreesClassifier**, and other classifiers
- **SMOTE** (Synthetic Minority Over-sampling Technique) for oversampling to handle class imbalance
- **Joblib** for saving and loading the trained model
- **Pandas** for data manipulation
- **Matplotlib**, **Seaborn** for data visualization

## Data Preprocessing

The project includes several steps for preparing the data before training the models:

1. **Handling Missing Values**: Ensuring no missing values are in the dataset.
2. **Feature Engineering**: Processing categorical and numerical features for model compatibility.
3. **Outlier Handling**: Using percentile capping to manage outliers.
4. **Normalization**: Scaling numerical features using **RobustScaler**.
5. **Data Splitting**: Splitting the data into training and testing sets (80% training, 20% testing).
6. **Handling Imbalanced Data**: Using SMOTE for oversampling the minority class to handle imbalance in the target variable.

## Model Building

The project utilizes various classifiers for building the prediction model, including:

- **Logistic Regression**
- **Ridge Classifier**
- **SVC** (Support Vector Classifier)
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **LightGBM**
- **CatBoost**
- **Extra Trees Classifier**

These models are trained, evaluated using metrics like **Log Loss**, **Recall**, and **F1 Score**, and the best-performing model is selected for further hyperparameter tuning.

## Model Evaluation

To evaluate model performance, the following metrics are used:
- **Log Loss**: The logarithmic loss function to evaluate the model's probability outputs.
- **Recall**: To check how well the model detects the positive class (loan repayment).
- **F1 Score**: To balance precision and recall.

## Hyperparameter Tuning

Hyperparameter tuning is performed using **GridSearchCV** to find the best model settings and improve performance. After performing grid search, the model with the best parameters is selected for final training.

### Steps for Hyperparameter Tuning:
1. **GridSearchCV**: Used for searching the best parameters across different options.
2. **Scoring**: The model is evaluated based on **Log Loss** (chosen metric for optimization).
3. **Cross-validation**: 5-fold cross-validation is used to reduce overfitting and ensure generalizability.

## Prerequisites
To run the code locally, you will need Python 3.8+ and pip installed. The following libraries are required:

- pandas
- numpy
- scikit-learn
- pycaret
- xgboost
- catboost
- lightgbm
- smote
- joblib
- matplotlib
- seaborn


## Conclusion
This project demonstrates the use of machine learning for predicting loan repayment based on various features. After handling class imbalance, feature preprocessing, and model training, the best model is selected and optimized using hyperparameter tuning to achieve the lowest log loss score. The final trained model can be used to predict loan repayment likelihood for new applicants.
