# Diabetes Risk Prediction Project

This project aims to predict the risk of diabetes using machine learning techniques. It leverages health survey data to train two classification models: Logistic Regression and Random Forest. The models are evaluated based on their accuracy, precision, recall, and F1-score.

## Objective

To build a binary classification system capable of predicting diabetes risk based on patient history and lifestyle data.

## Models Used

* **Logistic Regression**: A classical statistical model known for its interpretability.
* **Random Forest**: An ensemble model known for its performance in capturing complex patterns.

## Dataset

* Source: [Kaggle - Diabetes, Hypertension and Stroke Prediction](https://www.kaggle.com/datasets/prosperchuks/health-dataset)
* Rows: 70,692
* Features: 17 input features + 1 target variable (`Diabetes`)

## Features

The dataset includes features like age group, gender, BMI, physical activity, diet habits, smoking, alcohol use, and general health status.

## Methodology

1. **Data Preprocessing**: Checked for missing values, standardized numeric features using `StandardScaler`.
2. **Train-Test Split**: 70% training and 30% testing.
3. **Model Training**: Both Logistic Regression and Random Forest were trained and evaluated.
4. **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-Score.

## Results

* **Accuracy**: \~75% for both models
* **Random Forest** had higher recall in detecting diabetic cases (80%).
* **Logistic Regression** offered a balanced performance and better interpretability.

## Future Work

* Incorporate additional features (e.g., diet patterns, family history, sleep habits).
* Test advanced models such as XGBoost or LightGBM.
* Integrate into a real-time health monitoring application.

## License

This project is for academic and research purposes.

---

> This repository demonstrates the power of machine learning in health risk prediction and offers a reproducible workflow for developing classification models using real-world survey data.
