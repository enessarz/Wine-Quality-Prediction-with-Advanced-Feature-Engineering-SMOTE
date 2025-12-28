# 🍷 Wine Quality Prediction with Advanced Feature Engineering & SMOTE

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project aims to classify the quality of red wines (Low, Normal, High) based on their physicochemical properties. The dataset suffers from severe class imbalance, which was addressed using **SMOTE (Synthetic Minority Over-sampling Technique)** and class weighting strategies.

The model focuses on maximizing the detection of minority classes (Low and High quality) rather than just achieving high accuracy on the majority class.

## ⚙️ Key Techniques & Methodology

### 1. Data Cleaning & Outlier Handling
- **Winsorization (Quantile Capping):** Applied to columns like `residual sugar` and `chlorides` to cap extreme outliers at the 1st and 99th percentiles.
- **IQR Method:** Used for removing remaining outliers in other continuous features.

### 2. Feature Engineering (Domain Knowledge)
To improve model performance, I created interaction features based on chemical properties:
- **`total_acidity`**: Sum of fixed and volatile acidity.
- **`sugar_acidity_ratio`**: Interaction between residual sugar and fixed acidity.
- **`sulfur_ratio`**: Ratio of free sulfur dioxide to total sulfur dioxide.
- **`alcohol_sugar_interaction`**: Combined effect of alcohol and sugar content.

### 3. Data Transformation
- **Skewness Fix:** Applied `np.log1p` (Log transformation) to skewed features to normalize the distribution.
- **Scaling:** Used `StandardScaler` to standardize features before modeling.

### 4. Handling Imbalance
The original dataset was heavily biased towards "Normal" quality wines.
- **Strategy:** Applied **SMOTE** to the training set to generate synthetic samples for minority classes ("Low" and "High" quality).

## 🚀 Model & Results
**Algorithm:** Random Forest Classifier (Optimized via GridSearchCV)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **77%** |
| **Weighted F1-Score** | **0.78** |
| **Macro F1-Score** | **0.54** |

> **Note:** While the baseline accuracy was higher (85%), the initial model failed to detect "Low" quality wines (Recall: 0.00). After applying SMOTE and Feature Engineering, the model successfully captures minority classes, providing a more reliable and robust prediction.
