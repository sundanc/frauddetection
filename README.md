# Fraud Detection Model Using Ensemble Learning and Bayesian Optimization

## Overview
This project implements a fraud detection system using machine learning techniques to identify fraudulent transactions. The model leverages various ensemble classifiers, including XGBoost, LightGBM, and CatBoost, alongside a stacking classifier. Hyperparameters for each model are optimized using Bayesian Optimization. To address class imbalance in the dataset, SMOTE-ENN (Synthetic Minority Over-sampling Technique with Edited Nearest Neighbors) is applied for resampling.

## Key Features

### 1. Data Preprocessing
- **Label Encoding**: Categorical variables such as `merchant`, `category`, `first`, `last`, `gender`, and `street` are label encoded for machine learning compatibility.
- **Time Feature Extraction**: Transaction timestamps are converted into seconds since the earliest transaction to create a numerical time feature.
- **Standard Scaling**: The `amt` feature (transaction amount) is scaled using StandardScaler to normalize the values.
- **Missing Data Handling**: Missing values in the dataset are handled using imputation techniques.

### 2. Model Selection
- **XGBoost**: A powerful gradient boosting framework optimized for performance.
- **LightGBM**: A fast, distributed gradient boosting framework that supports large datasets.
- **CatBoost**: A gradient boosting algorithm designed to handle categorical data effectively.
- **Stacking Classifier**: Combines the predictions of multiple base models (XGBoost, LightGBM, CatBoost) and trains a meta-model (XGBoost) to improve overall performance.

### 3. Hyperparameter Optimization
- **Bayesian Optimization**: Used to tune the hyperparameters for each of the individual models (XGBoost, LightGBM, and CatBoost), ensuring optimal performance.
- The optimized parameters are used in the final ensemble model for better accuracy.

### 4. Class Imbalance Handling
- **SMOTE-ENN**: Synthetic Minority Over-sampling Technique with Edited Nearest Neighbors is used to address class imbalance by oversampling the minority class while cleaning noisy instances.

## Project Files
1. **fraudTrain.csv**: Training dataset containing transaction data.
2. **fraudTest.csv**: Test dataset used for evaluation.
3. **`datasets.py`**: Script for downloading the latest version of the dataset from Kaggle.
4. **Main Script**: Implements the fraud detection model with data preprocessing, model training, hyperparameter optimization, and evaluation.

## Usage

1. **Install Dependencies**:
   Install required libraries using `pip`:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost bayesian-optimization kagglehub
   ```

2. **Download Dataset**:
   Use the `datasets.py` script to download the dataset:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
   print("Path to dataset files:", path)
   ```

3. **Run the Model**:
   Execute the main script to preprocess the data, train the models, and evaluate their performance. The model will output the best hyperparameters for each classifier and the final classification results.

## Results
The model evaluates the performance of the stacked ensemble on the test dataset using the following metrics:
- **Classification Report**: Precision, Recall, F1-Score, and Support for each class.
- **ROC AUC Score**: Measures the model's ability to distinguish between fraudulent and non-fraudulent transactions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
