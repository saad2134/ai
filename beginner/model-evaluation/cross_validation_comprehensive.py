from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Define model
model = LogisticRegression(random_state=42, max_iter=1000)

# Define scoring metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Perform 5-fold cross-validation
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=True)

# Create results dataframe
results_df = pd.DataFrame({
    'Fold': range(1, 6),
    'Test Accuracy': cv_results['test_accuracy'],
    'Test Precision': cv_results['test_precision_macro'],
    'Test Recall': cv_results['test_recall_macro'],
    'Test F1': cv_results['test_f1_macro']
})

print("=== 5-Fold Cross-Validation Results ===")
print(results_df.round(4))
print("\n=== Average Scores ===")
print(results_df.mean().round(4))