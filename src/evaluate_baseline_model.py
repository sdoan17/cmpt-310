import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs('../reports', exist_ok=True)

X_test, y_test = joblib.load('../models/test_set.pkl')

dt_model = joblib.load('../models/decision_tree.pkl')
knn_model = joblib.load('../models/knn_baseline.pkl')

models = {
    'decision_tree' : dt_model,
    'knn' : knn_model
}

report_path = '../reports/base_models_report.txt'
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

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm.T, annot = True, fmt='d', cmap = "Blues",
                    xticklabels=['Actual_Dropout', 'Actual_No_Risk'],
                    yticklabels=['Predicted_Dropout', 'Predicted_No_Risk'])
        plt.title(f'{name.title()} Confusion Matrix')
        plt.savefig(f'../reports/{name}_confusion_matrix.png', bbox_inches = 'tight')
        plt.close()
   
print("Reports saved in ../reports/")