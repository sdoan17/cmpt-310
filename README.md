# 🎓 Student Dropout Prediction

## 📘 Introduction

Student dropout is a critical challenge faced by educational institutions worldwide. Identifying at-risk students early can allow for timely intervention and support, ultimately improving retention rates and academic outcomes. 

This project applies machine learning to predict whether a student is at risk of dropping out based on academic and behavioral data. The goal is to assist schools in allocating support resources more effectively and reducing dropout rates.

---

## 🎯 Project Objectives

- ✅ Classify students as `Dropout` or `No Risk`
- ✅ Evaluate multiple machine learning algorithms
- ✅ Tune hyperparameters using cross-validation
- ✅ Assess performance using metrics like accuracy, precision, recall, and F1-score
- ✅ Recommend a final model for deployment

---

## 💾 Dataset Description

- The dataset was obtained from a higher education institution and compiled from multiple sources covering various undergraduate programs (e.g., agronomy, education, nursing, management, journalism).
- It includes **student demographic, academic, and socio-economic features** known at the time of enrollment, along with academic outcomes after the first and second semesters.
- The dataset was originally formulated as a **three-category classification problem**, but for this project, we simplified it to a **binary classification**: `Dropout (1)` or `No Risk (0)`.
- There are **36 features** in total, consisting of **real, categorical, and integer** types.
- The dataset has **4,424 instances** in total.
- No missing values were present in the original dataset.
- Dataset source: [UCI Machine Learning Repository – Predict Students Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess?utm_source=chatgpt.com).

---

## 🛠️ Methodology

### 1. Preprocessing
- Cleaned missing values and encoded categorical variables
- Split the data into:
  - **Training set (80%)**
  - **Test set (20%)**

### 2. Model Training
- Two models were trained:
  - `DecisionTreeClassifier`
  - `KNeighborsClassifier`
- Applied **Stratified 5-Fold Cross-Validation** to maintain class balance during training

### 3. Features Used In Training
- Original Features (12):
  - Age at enrollment
  - Admission grade
  - Curricular units 1st sem (approved)
  - Curricular units 1st sem (grade)
  - Curricular units 2nd sem (approved)
  - Curricular units 2nd sem (grade)
  - Debtor
  - Tuition fees up to date
  - Scholarship holder
  - Displaced
  - Educational special needs
  - Gender
- Engineered Features (4):
  - overall_pass_rate
  - grade_diff
  - financial_risk
  - grade_x_passrate

### 4. Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold `StratifiedKFold` to optimize:
  - `max_depth`, `min_samples_split`, `min_samples_leaf` (Decision Tree)
  - `n_neighbors`, `weights`, `p` (KNN)

### 5. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- Macro and Weighted Averages

---

## 📊 Model Evaluation & Results

| Model               | Accuracy | F1-Score (Dropout) | F1-Score (No Risk) |
|--------------------|----------|--------------------|--------------------|
| Decision Tree (Tuned) | **0.84**   | **0.74**             | 0.89               |
| KNN (Tuned)           | 0.83     | 0.70                | **0.88**           |

### 🧠 Confusion Matrix Insights

- **Decision Tree** captured more actual dropouts (higher recall) but made slightly more false positives.
- **KNN** was more conservative, making fewer false dropout predictions but missing more true cases.

---

### 🔎 Final Verdict

- If your goal is to **catch as many dropouts as possible** → go with **Decision Tree Tuned**.
- If you care about **being conservative with dropout predictions** (i.e., fewer false alarms) → consider **KNN Tuned**.
- If you only care about **overall accuracy** — again, **Decision Tree wins**.

---


