from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load your train set
X_train, y_train = joblib.load('../models/train_set.pkl')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Decision Tree grid
dt_param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt, dt_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
dt_grid.fit(X_train, y_train)
print("DT best params:", dt_grid.best_params_)

# KNN grid
knn_param_grid = {
    'n_neighbors': [3,5,7,9],
    'weights': ['uniform','distance'],
    'p': [1,2]
}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
knn_grid.fit(X_train, y_train)
print("KNN best params:", knn_grid.best_params_)

# Save tuned models
joblib.dump(dt_grid.best_estimator_, '../models/decision_tree_tuned.pkl')
joblib.dump(knn_grid.best_estimator_, '../models/knn_tuned.pkl')
