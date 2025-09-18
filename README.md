# Iowa House Price Prediction: A Regression Analysis

> **Note:** The notebook in this repository is a clean version with all cell outputs cleared for fast rendering on GitHub.
>
> **➡️ [View the fully rendered notebook on NBViewer](https://nbviewer.org/github/shahrosek/iowa-house-price-prediction/blob/main/house-price-prediction-eda-and-feature-engineering.ipynb)**
>
> **➡️ [View the original notebook on Kaggle](https://www.kaggle.com/code/shahrosek/house-price-prediction-eda-and-feature-engineering/notebook)**

This repository contains the code and analysis for a comprehensive machine learning project focused on predicting residential home prices in Ames, Iowa. The goal was to apply advanced feature engineering and modeling techniques to achieve the highest possible accuracy on the well-known Kaggle dataset.

## 🎯 Objective
To build a robust regression model that accurately predicts the final sale price of a house based on its various quantitative and qualitative features.

## 📊 Dataset
The project utilizes the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) from Kaggle, which includes 79 explanatory variables describing every aspect of residential homes.

## ⚙️ Methodology & Technical Walkthrough
The project followed a structured data science workflow:

1.  **Data Cleaning & Preprocessing**:
    * Combined training and testing sets for consistent feature treatment.
    * Handled missing values (`NaN`) using strategic imputation (e.g., mode-filling for `LotFrontage`).
    * Cleaned and standardized categorical feature values.

2.  **Advanced Feature Engineering**:
    * **Target Ordinal Encoding**: The core of the feature engineering process. For 43 categorical features, each category was mapped to a numerical rank based on the median `SalePrice` for that category, embedding crucial price information directly into the features.
    * **Feature Scaling**: Analyzed the distribution of all numerical features and applied a dual strategy: `MinMaxScaler` (Normalization) for skewed data and `StandardScaler` (Standardization) for normally distributed data.
    * **Feature Selection**: Calculated a correlation matrix and programmatically removed features with high multicollinearity (>0.85) to improve model stability.

3.  **Model Training & Evaluation**:
    * Trained and compared two powerful gradient boosting models: **XGBoost** and **CatBoost**.
    * Used **Mean Absolute Percentage Error (MAPE)** as the primary metric to evaluate model performance on a validation set.

## 📈 Results & Outcome
The final **CatBoost Regressor** model demonstrated superior performance, achieving a **MAPE of 8.4%** on the validation set. The notebook generates a `submission.csv` file ready for the Kaggle competition.

## 🛠️ Tech Stack
* **Language**: `Python`
* **Libraries**: `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `XGBoost`, `CatBoost`
