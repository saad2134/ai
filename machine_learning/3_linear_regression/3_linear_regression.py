#1: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#2: Generating Random/Dummy Dataset
np.random.seed(42)
X = np.random.rand(50, 1) * 100
print(X)
Y = 3.5 * X + np.random.randn(50, 1) * 20

#3: Create & Train Linear Regression Mode
model = LinearRegression()
model.fit(X, Y)

#4: Predicting Y Values
Y_pred = model.predict(X)

#5: Visualizing the Regression Line
plt.figure(figsize=(8,6))
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')
plt.title("Linear Regression on Random Dataset")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
 
