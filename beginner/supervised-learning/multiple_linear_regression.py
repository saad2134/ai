import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Student performance with multiple features
data = {
    'study_hours': [10, 15, 20, 25, 30, 5, 12, 18, 22, 28],
    'attendance': [85, 90, 95, 98, 100, 70, 80, 88, 92, 96],
    'previous_score': [60, 65, 70, 75, 80, 55, 62, 68, 72, 78],
    'final_score': [65, 72, 80, 85, 90, 58, 70, 78, 82, 88]
}
df = pd.DataFrame(data)

X = df[['study_hours', 'attendance', 'previous_score']]
y = df['final_score']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Results
y_pred = model.predict(X_test)
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print(f"Intercept: {model.intercept_:.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")