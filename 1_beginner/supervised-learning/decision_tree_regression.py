import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Decision Tree for regression
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Train different depth trees
depths = [2, 4, 6]
plt.figure(figsize=(15, 5))

for i, depth in enumerate(depths):
    tree_reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree_reg.fit(X, y)
    
    # Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_pred = tree_reg.predict(X_test)
    
    # Plot
    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test, y_pred, color="cornflowerblue", label="prediction", linewidth=2)
    plt.title(f"Max Depth = {depth}")
    plt.legend()

plt.tight_layout()
plt.show()