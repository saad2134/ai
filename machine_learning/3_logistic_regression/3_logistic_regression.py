import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
data = pd.read_csv("../datasets/diabetes.csv")
print(data.info())
print("-------------------------------------------------")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
new_patient = [[2,120,70,25,100,30.5,0.5,35]]
prediction = model.predict(new_patient)
if prediction[0] == 1:
    print("\nNew Patient Prediction: Diabetic")
else:
    print("\nNew Patient Prediction: Not Diabetic")
plt.figure(figsize=(1005,10))
plot_tree(model, feature_names=X.columns,
          class_names=["No","Yes"], filled=True)
plt.show()