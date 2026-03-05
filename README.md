# Heart Disease Prediction Notebook

This notebook demonstrates a machine learning pipeline for predicting heart disease based on various health parameters. The process involves data loading, exploratory data analysis, model training using Logistic Regression, evaluation, and a simple predictive system.

## Table of Contents
1.  [Overview](#overview)
2.  [Dataset](#dataset)
3.  [Data Preprocessing and Exploration](#data-preprocessing-and-exploration)
4.  [Model Training](#model-training)
5.  [Model Evaluation](#model-evaluation)
6.  [Predictive System](#predictive-system)

## 1. Overview
This project aims to build a classification model to predict whether a person has heart disease (1) or not (0) based on their medical attributes. A Logistic Regression model is used for this purpose.

## 2. Dataset
The dataset `heart_disease_data.csv` contains various features related to heart health. Key columns include:
-   `age`: Age of the patient
-   `sex`: Sex of the patient (1 = male; 0 = female)
-   `cp`: Chest pain type (0-3)
-   `trestbps`: Resting blood pressure
-   `chol`: Serum cholestoral in mg/dl
-   `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
-   `restecg`: Resting electrocardiographic results
-   `thalach`: Maximum heart rate achieved
-   `exang`: Exercise induced angina (1 = yes; 0 = no)
-   `oldpeak`: ST depression induced by exercise relative to rest
-   `slope`: The slope of the peak exercise ST segment
-   `ca`: Number of major vessels (0-3) colored by flourosopy
-   `thal`: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
-   `target`: Prediction target (1 = heart disease; 0 = no heart disease)

**Dataset Shape**: (303 rows, 14 columns)

## 3. Data Preprocessing and Exploration
-   **Loading Data**: The dataset is loaded using pandas.
-   **Initial Inspection**: `heart_data.head()`, `heart_data.tail()`, `heart_data.shape`, `heart_data.info()`, `heart_data.isnull().sum()`, and `heart_data.describe()` were used to understand the data's structure, check for missing values, and get statistical summaries.
-   **Target Distribution**: The distribution of the `target` variable was checked using `heart_data['target'].value_counts()`, revealing a balanced dataset with 165 positive cases (heart disease) and 138 negative cases.
-   **Data Splitting**: The features (X) and target (Y) are separated. The data is then split into training (80%) and testing (20%) sets using `train_test_split` with `stratify=Y` to maintain the target distribution and `random_state=2` for reproducibility.

## 4. Model Training
-   A `LogisticRegression` model from `sklearn.linear_model` is initialized and trained on the `X_train` and `Y_train` data.

## 5. Model Evaluation
-   The trained model's performance is evaluated using `accuracy_score` on both the training and test datasets.
    -   **Training Data Accuracy**: 85.12%
    -   **Test Data Accuracy**: 81.97%
-   *Note*: A `ConvergenceWarning` was observed during training, suggesting that increasing `max_iter` or scaling the data might improve model convergence. This can be addressed in future iterations.

## 6. Predictive System
-   A simple predictive system is demonstrated where new input data (a tuple of 13 features) is fed to the trained model.
-   The model predicts whether the person has heart disease (0 or 1) based on the input features.
-   **Example Prediction**: For the input data `(62,0,0,160,164,0,0,145,0,6.2,0,3,3)`, the model predicted `0`, indicating that "The person does not have a heart disease".

