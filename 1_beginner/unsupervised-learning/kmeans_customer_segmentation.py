import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Create customer data
np.random.seed(42)
n_customers = 200
data = {
    'age': np.random.randint(18, 70, n_customers),
    'annual_income': np.random.randint(15, 150, n_customers) * 1000,
    'spending_score': np.random.randint(1, 100, n_customers)
}
df = pd.DataFrame(data)

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add clusters to dataframe
df['cluster'] = clusters

print("Customer Segments:")
print(df.groupby('cluster').mean())

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(df['annual_income'], df['spending_score'], c=df['cluster'], cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation using K-Means')
plt.colorbar(label='Cluster')
plt.show()