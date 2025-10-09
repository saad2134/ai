import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Create Titanic-like dataset
data = {
    'pclass': [1, 1, 2, 2, 3, 3, 1, 2, 3, 3],
    'sex': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],  # 0=female, 1=male
    'age': [29, 2, 25, 30, 22, 28, 35, 45, 20, 18],
    'sibsp': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],  # siblings/spouse
    'survived': [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]  # target
}
df = pd.DataFrame(data)

X = df[['pclass', 'sex', 'age', 'sibsp']]
y = df['survived']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Evaluate
y_pred = tree.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Visualize tree
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, class_names=['Died', 'Survived'], filled=True)
plt.show()