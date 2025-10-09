import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

class ActivationFunctions:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to prevent overflow
    
    @staticmethod
    def sigmoid_derivative(x):
        return ActivationFunctions.sigmoid(x) * (1 - ActivationFunctions.sigmoid(x))
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class LossFunctions:
    """Collection of loss functions and their derivatives"""
    
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

class Perceptron:
    """Single Layer Perceptron"""
    
    def __init__(self, input_size, learning_rate=0.01, activation='sigmoid'):
        self.weights = np.random.randn(input_size) * 0.1
        self.bias = np.random.randn() * 0.1
        self.learning_rate = learning_rate
        self.activation_name = activation
        
        # Set activation function
        if activation == 'sigmoid':
            self.activation = ActivationFunctions.sigmoid
            self.activation_derivative = ActivationFunctions.sigmoid_derivative
        elif activation == 'relu':
            self.activation = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        else:
            raise ValueError("Unsupported activation function")
    
    def forward(self, X):
        self.z = np.dot(X, self.weights) + self.bias
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, X, y, output):
        # Calculate gradients
        m = X.shape[0]
        dz = output - y  # For binary classification with sigmoid
        
        dw = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz)
        
        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = LossFunctions.binary_cross_entropy(y, output)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                accuracy = accuracy_score(y, (output > 0.5).astype(int))
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return losses

class MLP:
    """Multi-Layer Perceptron from Scratch"""
    
    def __init__(self, layer_sizes, learning_rate=0.1, activation='relu', output_activation='sigmoid'):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.output_activation_name = output_activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU, Xavier for sigmoid/tanh
            if activation == 'relu':
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])
            
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Set activation functions
        self.set_activation_functions(activation, output_activation)
    
    def set_activation_functions(self, hidden_activation, output_activation):
        """Set activation functions for hidden and output layers"""
        activation_map = {
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative)
        }
        
        self.activation, self.activation_derivative = activation_map[hidden_activation]
        
        if output_activation == 'softmax':
            self.output_activation = ActivationFunctions.softmax
            self.output_activation_derivative = None  # Handled separately in cross-entropy
        else:
            self.output_activation, self.output_activation_derivative = activation_map[output_activation]
    
    def forward(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        self.z_values = []
        
        current_activation = X
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_activation = self.activation(z)
            self.activations.append(current_activation)
        
        # Output layer
        z_output = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        
        if self.output_activation_name == 'softmax':
            output_activation = self.output_activation(z_output)
        else:
            output_activation = self.output_activation(z_output)
        
        self.activations.append(output_activation)
        
        return output_activation
    
    def backward(self, X, y):
        """Backward pass (backpropagation)"""
        m = X.shape[0]
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        if self.output_activation_name == 'softmax':
            # Softmax + cross-entropy derivative simplifies to (y_pred - y_true)
            dz = self.activations[-1] - y
        else:
            if self.output_activation_name == 'sigmoid':
                dz = (self.activations[-1] - y) * self.output_activation_derivative(self.z_values[-1])
            else:
                dz = (self.activations[-1] - y)  # For linear output
        
        gradients_w[-1] = (1/m) * np.dot(self.activations[-2].T, dz)
        gradients_b[-1] = (1/m) * np.sum(dz, axis=0, keepdims=True)
        
        # Backpropagate through hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(dz, self.weights[l+1].T) * self.activation_derivative(self.z_values[l])
            gradients_w[l] = (1/m) * np.dot(self.activations[l].T, dz)
            gradients_b[l] = (1/m) * np.sum(dz, axis=0, keepdims=True)
        
        return gradients_w, gradients_b
    
    def update_parameters(self, gradients_w, gradients_b):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def compute_loss(self, y_true, y_pred):
        """Compute appropriate loss based on output activation"""
        if self.output_activation_name == 'softmax':
            return LossFunctions.cross_entropy(y_true, y_pred)
        elif self.output_activation_name == 'sigmoid':
            return LossFunctions.binary_cross_entropy(y_true, y_pred)
        else:
            return LossFunctions.mse(y_true, y_pred)
    
    def train(self, X, y, epochs=1000, verbose=True):
        """Train the neural network"""
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            
            # Compute accuracy
            if self.output_activation_name == 'softmax':
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y, axis=1)
                accuracy = np.mean(y_pred_classes == y_true_classes)
            else:
                accuracy = accuracy_score(y, (y_pred > 0.5).astype(int))
            accuracies.append(accuracy)
            
            # Backward pass and update
            gradients_w, gradients_b = self.backward(X, y)
            self.update_parameters(gradients_w, gradients_b)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return losses, accuracies
    
    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward(X)
        if self.output_activation_name == 'softmax':
            return np.argmax(y_pred, axis=1)
        else:
            return (y_pred > 0.5).astype(int)

def solve_xor_problem():
    """Solve the XOR problem with MLP"""
    print("=== Solving XOR Problem with MLP ===")
    
    # XOR dataset
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    print("XOR Dataset:")
    for i in range(len(X_xor)):
        print(f"Input: {X_xor[i]}, Target: {y_xor[i][0]}")
    
    # Create MLP with 2-4-1 architecture
    mlp = MLP(layer_sizes=[2, 4, 1], learning_rate=0.5, activation='tanh', output_activation='sigmoid')
    
    # Train the network
    losses, accuracies = mlp.train(X_xor, y_xor, epochs=5000, verbose=True)
    
    # Test the network
    print("\n=== XOR Problem Results ===")
    predictions = mlp.predict(X_xor)
    probabilities = mlp.forward(X_xor)
    
    for i in range(len(X_xor)):
        print(f"Input: {X_xor[i]}, Target: {y_xor[i][0]}, "
              f"Predicted: {predictions[i][0]}, Probability: {probabilities[i][0]:.4f}")
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return mlp

def visualize_activation_functions():
    """Visualize different activation functions"""
    x = np.linspace(-5, 5, 100)
    
    activations = {
        'Sigmoid': ActivationFunctions.sigmoid(x),
        'ReLU': ActivationFunctions.relu(x),
        'Tanh': ActivationFunctions.tanh(x)
    }
    
    derivatives = {
        'Sigmoid': ActivationFunctions.sigmoid_derivative(x),
        'ReLU': ActivationFunctions.relu_derivative(x),
        'Tanh': ActivationFunctions.tanh_derivative(x)
    }
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, activation in activations.items():
        plt.plot(x, activation, label=name, linewidth=2)
    plt.title('Activation Functions')
    plt.xlabel('x')
    plt.ylabel('Activation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for name, derivative in derivatives.items():
        plt.plot(x, derivative, label=name, linewidth=2)
    plt.title('Activation Function Derivatives')
    plt.xlabel('x')
    plt.ylabel('Derivative')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_on_synthetic_data():
    """Test MLP on synthetic classification data"""
    print("\n=== Testing MLP on Synthetic Data ===")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, 
                              n_informative=8, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape y for consistency
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Create and train MLP
    mlp = MLP(layer_sizes=[10, 16, 8, 1], learning_rate=0.1, 
              activation='relu', output_activation='sigmoid')
    
    losses, accuracies = mlp.train(X_train, y_train, epochs=1000, verbose=True)
    
    # Evaluate on test set
    test_predictions = mlp.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return mlp, test_accuracy

if __name__ == "__main__":
    # 1. Visualize activation functions
    visualize_activation_functions()
    
    # 2. Solve XOR problem
    xor_mlp = solve_xor_problem()
    
    # 3. Test on synthetic data
    synthetic_mlp, test_acc = test_on_synthetic_data()