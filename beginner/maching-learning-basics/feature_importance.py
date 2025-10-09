from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Get feature importance
importance = model.feature_importances_
feature_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("Top 5 most important features:")
print(feature_imp_df.head())

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_imp_df['feature'][:10], feature_imp_df['importance'][:10])
plt.title('Top 10 Feature Importance')
plt.show()