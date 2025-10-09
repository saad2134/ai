from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree hyperparameter tuning
tree_params = {
    'max_depth': [2, 3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_params, cv=5, scoring='accuracy')
tree_grid.fit(X_train, y_train)

print("Best Decision Tree Parameters:", tree_grid.best_params_)
print(f"Best CV Score: {tree_grid.best_score_:.3f}")

# KNN hyperparameter tuning
knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)

print("\nBest KNN Parameters:", knn_grid.best_params_)
print(f"Best CV Score: {knn_grid.best_score_:.3f}")

# Test best models
best_tree = tree_grid.best_estimator_
best_knn = knn_grid.best_estimator_

print(f"\nTest Accuracy - Decision Tree: {accuracy_score(y_test, best_tree.predict(X_test)):.3f}")
print(f"Test Accuracy - KNN: {accuracy_score(y_test, best_knn.predict(X_test)):.3f}")