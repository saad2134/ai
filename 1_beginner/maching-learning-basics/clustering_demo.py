from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create sample data with clusters
X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("K-Means Clustering")
plt.show()