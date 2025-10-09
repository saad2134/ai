import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Student data: study hours vs exam scores
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
exam_scores = np.array([50, 55, 65, 70, 75, 80, 85, 88, 90, 92])

# Create and train model
model = LinearRegression()
model.fit(study_hours, exam_scores)

# Predict
predicted_scores = model.predict(study_hours)

# Results
print(f"Equation: Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Hours")
print(f"R² Score: {r2_score(exam_scores, predicted_scores):.3f}")

# Plot
plt.scatter(study_hours, exam_scores, color='blue')
plt.plot(study_hours, predicted_scores, color='red')
plt.title('Study Hours vs Exam Scores')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.show()