from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create sample data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Try different values of k
k_values = [2, 3, 4, 5, 6]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    axes[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   marker='x', s=200, linewidths=3, color='red')
    axes[i].set_title(f'K-Means with k={k}\nInertia: {kmeans.inertia_:.1f}')
    axes[i].set_xlabel('Feature 1')
    axes[i].set_ylabel('Feature 2')

# Remove empty subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.show()