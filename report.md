**Diabetes Risk Prediction Using Logistic Regression and Random Forest**

## 1. Project Domain

### Background

Diabetes is a chronic disease and one of the leading causes of death worldwide. It often goes undetected in its early stages due to subtle symptoms. Therefore, early detection of diabetes risk is crucial to prevent long-term complications. Machine learning technology offers a promising approach to assist early detection using health history and lifestyle data.

Machine learning-based predictive models have been widely researched and proven effective in accurately and rapidly identifying potential diabetes risks. Leveraging available data, these models can serve as decision support systems in the healthcare domain.

### Why This Problem Matters

* Supports earlier diagnosis, allowing patients to intervene promptly.
* Provides data-driven decision-making support for medical professionals.
* Improves efficiency in diabetes management and complication prevention at the population level.

### References

* Kaur, H. and Kumari, V. (2022), "Predictive modelling and analytics for diabetes using a machine learning approach", *Applied Computing and Informatics*, Vol. 18 No. 1/2, pp. 90–100. [https://doi.org/10.1016/j.aci.2018.12.004](https://doi.org/10.1016/j.aci.2018.12.004)
* Modak, S.K.S., Jha, V.K. (2024). "Diabetes prediction model using machine learning techniques." *Multimedia Tools and Applications*, 83, 38523–38549. [https://doi.org/10.1007/s11042-023-16745-4](https://doi.org/10.1007/s11042-023-16745-4)

## 2. Business Understanding

### Problem Statement

How can diabetes risk be predicted based on patient history and lifestyle data?

### Goals

To build a high-accuracy diabetes classification model using machine learning that supports faster and more efficient early detection and prevention.

### Solution Statement

To achieve this goal, two machine learning models are built and compared:

1. **Logistic Regression (LR)**

   * A classical statistical model widely used for binary classification.
   * Highly interpretable and efficient as a baseline.
   * Provides easily understandable insights into the effect of each feature on diabetes risk.

2. **Random Forest Classifier (RF)**

   * A more complex ensemble model based on decision trees.
   * Can handle nonlinear relationships and feature interactions.
   * Expected to perform better with parameter tuning and feature selection.

## 3. Data Understanding

### Dataset

* Name: `diabetes_data.csv`
* Source: [Diabetes, Hypertension and Stroke Prediction](https://www.kaggle.com/datasets/prosperchuks/health-dataset)
* Samples: 70,692 rows
* Data types: Categorical and continuous numeric
* Target variable: `Diabetes` (0 = No, 1 = Yes)

### Dataset Description

This dataset contains public health survey data covering various aspects such as age, body mass index (BMI), high blood pressure, smoking habits, fruit/vegetable consumption, and physical activity. The objective is to predict whether a person is at risk of diabetes based on these variables.

### Dataset Statistics

* Total rows: 70,692, 18 columns (including target)
* No missing values
* Target distribution:

  * Class 0 (non-diabetic): 50.0%
  * Class 1 (diabetic): 50.0%
* Average BMI: 29.8
* Most common age group: 60–64 years (code 9)
* 56% of respondents have high blood pressure
* 52% have high cholesterol

### Dataset Features:

| Feature              | Description                                             |
| -------------------- | ------------------------------------------------------- |
| Age                  | Age category (1 = 18–24, ..., 13 = ≥80)                 |
| Sex                  | Gender (0 = female, 1 = male)                           |
| HighChol             | High cholesterol (1 = yes, 0 = no)                      |
| CholCheck            | Cholesterol check in the past 5 years (1 = yes, 0 = no) |
| BMI                  | Body Mass Index                                         |
| Smoker               | Smoked ≥100 cigarettes in lifetime (1 = yes, 0 = no)    |
| HeartDiseaseorAttack | History of heart attack (1 = yes, 0 = no)               |
| PhysActivity         | Physical activity in the past 30 days (1 = yes, 0 = no) |
| Fruits               | Daily fruit consumption (1 = yes, 0 = no)               |
| Veggies              | Daily vegetable consumption (1 = yes, 0 = no)           |
| HvyAlcoholConsump    | Heavy alcohol consumption (1 = yes, 0 = no)             |
| GenHlth              | General health rating (1 = excellent, 5 = very poor)    |
| MentHlth             | Days with mental health issues (last 30 days)           |
| PhysHlth             | Days with physical health issues (last 30 days)         |
| DiffWalk             | Difficulty walking (1 = yes, 0 = no)                    |
| Stroke               | Stroke history (1 = yes, 0 = no)                        |
| HighBP               | High blood pressure (1 = yes, 0 = no)                   |
| Diabetes             | Classification target (1 = diabetic, 0 = non-diabetic)  |

### Exploratory Data Analysis (EDA)

Conducted steps:

* Visualized target distribution (pie and bar charts)
* Analyzed age and gender distributions
* Correlation between numerical features and diabetes
* Descriptive statistics (mean, median, mode) for main features

## 4. Data Preparation

### Data Cleaning

* No missing values, no imputation needed.
* No duplicate records.
* Extreme values in `BMI`, `MentHlth`, and `PhysHlth` retained to reflect actual population conditions.
* All data are numeric, so no additional encoding needed.

### Standardization and Data Splitting

* Standardized using `StandardScaler` to normalize feature scales (mean = 0, std = 1).
* Split into training and testing sets (70:30 ratio).

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
```

## 5. Modeling

### Model Approach

This project addresses a binary classification problem: determining whether a person is at risk of diabetes based on medical and lifestyle attributes. Two machine learning models are used: Logistic Regression and Random Forest.

### 1. Logistic Regression (LR)

* **Description:** Logistic Regression is a classical statistical model used for binary classification. It models the probability of the target class using a logistic function of the linear combination of features.
* **Parameter:**

  * `class_weight='balanced'`: to address class imbalance.
* **Advantages:**

  * Fast and efficient for medium-sized datasets.
  * Easily interpretable, useful in medical contexts.
* **Disadvantages:**

  * Less capable of capturing non-linear feature relationships.
  * Sensitive to multicollinearity.

### 2. Random Forest (RF)

* **Description:** Random Forest is an ensemble algorithm that combines multiple decision trees to increase accuracy and reduce overfitting.
* **Parameters Used:**

  * `n_estimators=200`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2`
  * `class_weight='balanced'`, `random_state=50`
* **Advantages:**

  * Can handle non-linear and complex feature interactions.
  * More robust to overfitting than a single decision tree.
* **Disadvantages:**

  * Less transparent in interpretation compared to linear models.
  * Higher computational cost.

### Model Selection

After training and evaluating both models, Random Forest showed better performance in terms of accuracy and F1-score. It was better able to capture complex patterns in the data, especially non-linear feature interactions.

Thus, **Random Forest is chosen as the best model** in this project for its superior predictive performance, despite lower interpretability.

## 6. Evaluation

### Evaluation Metrics:

Models are evaluated using a confusion matrix and classification report to assess prediction performance on test data.

Metrics used:

* **Accuracy** — proportion of correct predictions
* **Precision** — proportion of correctly predicted positives (diabetes)
* **Recall** — model's ability to detect all diabetes cases
* **F1-Score** — harmonic mean of precision and recall

### Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions made by the classification model:

* **True Positive (TP):** Correctly predicted diabetes cases
* **True Negative (TN):** Correctly predicted non-diabetes cases
* **False Positive (FP):** Non-diabetes cases incorrectly predicted as diabetes
* **False Negative (FN):** Missed diabetes cases

This matrix provides insights into model errors across classes and serves as a basis for calculating precision, recall, and F1-score.

### Logistic Regression:

| Class        | Precision | Recall | F1-Score | Support   |
| ------------ | --------- | ------ | -------- | --------- |
| Non-Diabetic | 0.76      | 0.72   | 0.74     | 10601     |
| Diabetic     | 0.74      | 0.77   | 0.75     | 10607     |
| **Accuracy** |           |        | **0.75** | **21208** |
| Macro Avg    | 0.75      | 0.75   | 0.75     | 21208     |
| Weighted Avg | 0.75      | 0.75   | 0.75     | 21208     |

### Random Forest:

| Class        | Precision | Recall | F1-Score | Support   |
| ------------ | --------- | ------ | -------- | --------- |
| Non-Diabetic | 0.78      | 0.70   | 0.74     | 10601     |
| Diabetic     | 0.73      | 0.80   | 0.76     | 10607     |
| **Accuracy** |           |        | **0.75** | **21208** |
| Macro Avg    | 0.75      | 0.75   | 0.75     | 21208     |
| Weighted Avg | 0.75      | 0.75   | 0.75     | 21208     |

Both models demonstrated balanced performance with around 75% accuracy. Logistic Regression yielded balanced predictions across classes, while Random Forest showed higher recall for diabetic cases, making it more effective in detecting positives.

## 7. Conclusion

Diabetes risk classification models using Logistic Regression and Random Forest were successfully built and evaluated on health survey data.

Key points:

* Both models achieved **\~75% accuracy**, sufficient for binary classification.
* **Logistic Regression** provided stable and balanced results across classes.
* **Random Forest** excelled in detecting diabetes cases with up to 80% recall.
* Evaluation via confusion matrix showed reliability, with distinct strengths in each approach.
* This project confirms that basic health indicators such as blood pressure, BMI, and age can be used to build an effective early screening system for diabetes.

### Future Development:

* Include additional features like family history, diet, and sleep habits to enhance prediction.
* Perform further hyperparameter tuning for model optimization.
* Experiment with models like XGBoost or LightGBM for potential accuracy improvements.

This model can serve as a foundation for data-driven early detection systems in healthcare services or public health campaigns.
