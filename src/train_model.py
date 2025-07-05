# Script for training decision tree and KNN models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv("../data/processed/clean_data.csv")

#Binary target encoding: 0 = Dropout, 1 = No Risk
data['Target'] = data['Target'].map({0: 0, 1: 1, 2: 1})


# 4. Add engineered features
data['overall_pass_rate'] = (
    data['Curricular units 1st sem (approved)'] +
    data['Curricular units 2nd sem (approved)']
) / (
    data['Curricular units 1st sem (enrolled)'] +
    data['Curricular units 2nd sem (enrolled)']
)
data['grade_diff'] = (
    data['Curricular units 2nd sem (grade)'] -
    data['Curricular units 1st sem (grade)']
)
data['financial_risk'] = data['Debtor'] - data['Scholarship holder']
data['grade_x_passrate'] = (
    data['Curricular units 1st sem (grade)'] *
    data['Curricular units 1st sem (pass rate)']
)

features = [
    'Age at enrollment',
    'Admission grade',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Debtor',
    'Tuition fees up to date',
    'Scholarship holder',
    'Displaced',
    'Educational special needs',
    'Gender'
]

engineered_features = [
    'overall_pass_rate',
    'grade_diff',
    'financial_risk',
    'grade_x_passrate'
]

all_features = features + engineered_features

X = data[all_features]
y = data['Target']

# One-hot encoding
X = pd.get_dummies(X, drop_first = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

# Save the test set
joblib.dump((X_test,y_test), '../models/test_set.pkl')
tree = DecisionTreeClassifier(random_state = 10)
tree.fit(X_train, y_train)
joblib.dump(tree, '../models/decision_tree.pkl')

y_pred = tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plt.figure(figsize=(20, 10))
# plot_tree(tree, filled=True, feature_names=X.columns, class_names=['Dropout', 'No Risk'])
# plt.title("Decision Tree for RiskRadar")
# plt.show()
