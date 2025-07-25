import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

import joblib

# 1) Load the train_test set
X_train, y_train = joblib.load('../models/train_set.pkl')
X_test, y_test = joblib.load('../models/test_set.pkl')

# 2) Train the KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
joblib.dump(knn, '../models/knn_baseline.pkl')

# 3) Evaluate
y_pred = knn.predict(X_test)
print("KNN Baseline Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Dropout','No Risk']))