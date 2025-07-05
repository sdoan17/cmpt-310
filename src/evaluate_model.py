import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs('../reports', exist_ok=True)

X_test, y_test = joblib.load('../models/test_set.pkl')

dt_model = joblib.load('../models/decision_tree.pkl')

models = {
    'decision_tree' : dt_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=["Dropout", "No Risk"]
    )
    with open(f'../reports/{name}_classification_report.txt', 'w') as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm.T, annot = True, fmt='d', cmap = "Blues",
                xticklabels=['Actual_Dropout', 'Actual_No_Risk'],
                yticklabels=['Predicted_Dropout', 'Predicted_No_Risk'])
    plt.title(f'{name.title()} Confusion Matrix')
    plt.savefig(f'../reports/{name}_confusion_matrix.png', bbox_inches = 'tight')
    plt.close()

print("Reports saved in ../reports/")