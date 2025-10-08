# Basic Imports
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

# Initialize and train Multi-layer Feed Forward Neural Network / Multi-Layer Perceptron Classifier
mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, random_state=0)
mlp.fit(X_train, y_train)

# Calculate and print accuracy
print("Accuracy:", accuracy_score(y_test, mlp.predict(X_test)))



# Visualize the decision tree
import matplotlib.pyplot as plt
plt.plot(mlp.loss_curve_)
plt.title('Training Loss')
plt.show()