from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load dataset with many features
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"Original shape: {X.shape}")

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for different numbers of components
n_components_range = range(1, 11)
explained_variances = []

for n in n_components_range:
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    explained_variances.append(sum(pca.explained_variance_ratio_))

# Plot explained variance vs number of components
plt.figure(figsize=(8, 5))
plt.plot(n_components_range, explained_variances, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Explained Variance vs Number of Components')
plt.grid(True)
plt.show()

# Show how many components needed for 95% variance
pca_full = PCA().fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")