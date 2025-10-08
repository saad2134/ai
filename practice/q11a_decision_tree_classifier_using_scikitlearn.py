# Basic Imports
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic classification dataset
X, y = make_classification(
    n_samples=100,
    n_features=4,
    n_classes=2,
    random_state=0
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=1
)

# Initialize and train Decision Tree classifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Calculate and print accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



# Visualize the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plot_tree(
    clf, 
    filled=True, 
    feature_names=[f'Feature_{i}' for i in range(4)]
)
plt.title("Decision Tree Classifier")
plt.show()