# Decision Tress using Pima Indian Diabetic Dataset

# 0: Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1: Load Dataset
path = "../datasets/diabetes.csv" 
data = pd.read_csv(path)
print(data.info())
print("---------------------------------------")

# 2: Seperate Features & Target
X = data.drop("Outcome", axis=1) #Features
y = data["Outcome"] # Target (0 = No Diabetes, 1= Diabetes)

# 3: Split into Train & Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4: Build DecisionTree Model
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)

# 5: Predict on Test Data
y_pred = model.predict(X_test)

# 6: Evaluate the Model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print('--------------------')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7: Classify a NEW Patient
# Features Order:
# Pregnancies Glucose BloodPressure SkinThickness Insulin BMI DiabetesPedigreeFunction Age Outcome

new_patient = [[2, 120, 70, 25, 100, 30.5, 0.5, 35]]
predicition = model.predict(new_patient)
if predicition[0] == 1:
    print("\nNew Patient Prediction: Diabetic")
else:
    print("\nNew Patient Prediciton: Not Diabetic")

plt.figure(figsize=(15,20))
plot_tree(model, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()
