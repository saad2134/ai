#1: Imports
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#2: Load the dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

#3: Seperate features & target
X = df.drop(columns=['MedHouseVal'])
y = df["MedHouseVal"]

#4: Split data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


#5: Create & Train LinearRegression Model
model = LinearRegression()
model.fit(X_train, y_train)

#6: Make Predictions
y_pred = model.predict(X_test)
print(y_pred)
print("----------------------------------------------------")

#7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE): ", mse)
print("R² Score:", r2)
print("\nModel Coefficients:\n", model.coef_)
print("\nIntercept: ", model.intercept_)
