from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define comprehensive scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall', 
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Perform comprehensive cross-validation
model = RandomForestClassifier(random_state=42)
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=True)

# Create detailed results table
results_summary = {}
for metric in scoring.keys():
    test_scores = cv_results[f'test_{metric}']
    train_scores = cv_results[f'train_{metric}']
    
    results_summary[metric] = {
        'Test Mean': np.mean(test_scores),
        'Test Std': np.std(test_scores),
        'Train Mean': np.mean(train_scores),
        'Train Std': np.std(train_scores),
        'Overfitting Gap': np.mean(train_scores) - np.mean(test_scores)
    }

# Convert to DataFrame
results_df = pd.DataFrame(results_summary).T
print("=== Comprehensive 5-Fold Cross-Validation Results ===")
print(results_df.round(4))

# Final evaluation on test set
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

final_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_proba)
}

print("\n=== Final Test Set Performance ===")
for metric, value in final_metrics.items():
    print(f"{metric}: {value:.4f}")

# Check for overfitting
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nTraining Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")
print(f"Overfitting Gap: {train_score - test_score:.4f}")