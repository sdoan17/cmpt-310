import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Load test set
X_test, y_test = joblib.load('../models/test_set.pkl')

# Load tune model
dt_tuned = joblib.load('../models/decision_tree_tuned.pkl')
knn_tuned = joblib.load('../models/knn_tuned.pkl')

models = {
    "Decision_Tree_Tuned" : dt_tuned,
    "KNN_Tuned" : knn_tuned
}

report_path = '../reports/tuned_models_report.txt'
with open(report_path, 'w') as f:
    for name, model in models.items():
        # Classification Report
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=["Dropout", "No Risk"]
        )
        f.write(f"=== {name} ===\n")
        f.write(report)
        f.write("\n\n")

print(f"Combined report saved to {report_path}")