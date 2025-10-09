from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
feature_names = cancer.feature_names

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Create component analysis
components_df = pd.DataFrame(pca.components_.T,
                           columns=[f'PC{i+1}' for i in range(len(feature_names))],
                           index=feature_names)

print("Top features for first two principal components:")
print("PC1 (Most important):", components_df['PC1'].abs().nlargest(5).index.tolist())
print("PC2 (Second most important):", components_df['PC2'].abs().nlargest(5).index.tolist())

# Plot feature importance in components
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
components_df['PC1'].abs().nlargest(10).plot(kind='bar')
plt.title('Top 10 Features for PC1')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
components_df['PC2'].abs().nlargest(10).plot(kind='bar')
plt.title('Top 10 Features for PC2')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()