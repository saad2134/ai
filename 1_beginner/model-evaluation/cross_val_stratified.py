from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

# Load imbalanced dataset (we'll use iris but demonstrate the concept)
iris = load_iris()
X, y = iris.data, iris.target

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Compare different cross-validation strategies
cv_strategies = {
    'KFold (5-fold)': KFold(n_splits=5, shuffle=True, random_state=42),
    'StratifiedKFold (5-fold)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
}

results = {}
for model_name, model in models.items():
    model_results = {}
    for cv_name, cv in cv_strategies.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        model_results[cv_name] = {
            'Mean Accuracy': np.mean(scores),
            'Std Accuracy': np.std(scores),
            'All Scores': scores
        }
    results[model_name] = model_results

# Display results
for model_name, model_result in results.items():
    print(f"\n=== {model_name} ===")
    for cv_name, cv_result in model_result.items():
        print(f"{cv_name}: {cv_result['Mean Accuracy']:.4f} (±{cv_result['Std Accuracy']:.4f})")