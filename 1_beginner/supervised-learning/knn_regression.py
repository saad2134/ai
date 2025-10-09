import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# KNN for regression - predicting house prices
size_sqft = np.array([750, 1000, 1250, 1500, 1750, 2000, 2250, 2500])
price = np.array([150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000])

# Reshape for sklearn
X = size_sqft.reshape(-1, 1)
y = price

# Train KNN regressor
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X, y)

# Predict
X_test = np.array([1200, 1800, 2200]).reshape(-1, 1)
predictions = knn_reg.predict(X_test)

print("KNN Regression Predictions:")
for size, pred in zip(X_test.flatten(), predictions):
    print(f" {size} sqft -> ${pred:,.0f}")

# Plot
plt.scatter(X, y, color='blue', label='Actual')
X_plot = np.linspace(700, 2600, 100).reshape(-1, 1)
y_plot = knn_reg.predict(X_plot)
plt.plot(X_plot, y_plot, color='red', label='KNN Regression')
plt.title('KNN Regression: House Size vs Price')
plt.xlabel('Size (sqft)')
plt.ylabel('Price ($)')
plt.legend()
plt.show()