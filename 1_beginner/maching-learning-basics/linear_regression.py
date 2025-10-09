from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample data: square footage vs house price
X = np.array([750, 1000, 1250, 1500, 1750, 2000]).reshape(-1, 1)  # sq ft
y = np.array([150000, 200000, 250000, 300000, 350000, 400000])    # price

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict
predictions = model.predict([[1800], [2200]])
print(f"Predicted prices: {predictions}")
print(f"R² Score: {model.score(X, y):.3f}")