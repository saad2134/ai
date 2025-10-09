from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Apply PCA first
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Then apply K-Means on reduced data
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# True labels
sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
ax1.set_title('True Labels (After PCA)')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
plt.colorbar(sc1, ax=ax1, label='True Species')

# K-Means clusters
sc2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           marker='x', s=200, linewidths=3, color='red')
ax2.set_title('K-Means Clusters (After PCA)')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
plt.colorbar(sc2, ax=ax2, label='Cluster')

plt.tight_layout()
plt.show()

print(f"PCA explained variance: {sum(pca.explained_variance_ratio_):.3f}")