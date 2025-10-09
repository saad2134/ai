import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class MNISTMLP:
    """MLP for MNIST classification using Keras/TensorFlow"""
    
    def __init__(self, hidden_layers=[128, 64], activation='relu', 
                 output_activation='softmax', learning_rate=0.001):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self, input_shape, num_classes):
        """Build the MLP model architecture"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Flatten layer (for image data)
        model.add(layers.Flatten())
        
        # Hidden layers
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation=self.activation))
            # Add batch normalization for better training
            model.add(layers.BatchNormalization())
            # Add dropout for regularization
            model.add(layers.Dropout(0.3))
        
        # Output layer
        model.add(layers.Dense(num_classes, activation=self.output_activation))
        
        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        if num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        return model
    
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        
        # Load MNIST dataset
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        # Normalize pixel values to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Reshape data if needed (MLP expects flattened input)
        # We'll use Flatten layer in the model instead
        
        return (X_train, y_train), (X_test, y_test)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=10, batch_size=128, verbose=1):
        """Train the MLP model"""
        print("Training MLP model...")
        
        if X_val is None or y_val is None:
            # Use 20% of training data for validation
            split_idx = int(0.8 * len(X_train))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
            ]
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        print("Evaluating model on test data...")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))
        
        return test_loss, test_accuracy, y_pred_classes
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training & validation loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training & validation accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    def visualize_predictions(self, X_test, y_test, num_samples=10):
        """Visualize some test predictions"""
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Select random samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            axes[i].imshow(X_test[idx], cmap='gray')
            axes[i].set_title(f'True: {y_test[idx]}, Pred: {y_pred_classes[idx]}')
            axes[i].axis('off')
            
            # Color title based on correctness
            if y_test[idx] == y_pred_classes[idx]:
                axes[i].title.set_color('green')
            else:
                axes[i].title.set_color('red')
        
        plt.tight_layout()
        plt.show()

def experiment_with_architectures():
    """Experiment with different MLP architectures"""
    # Load data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    architectures = [
        ([128], 'Simple MLP (128)'),
        ([256, 128], 'Deep MLP (256-128)'),
        ([512, 256, 128], 'Deeper MLP (512-256-128)'),
        ([128, 128, 128], 'Wide MLP (128-128-128)')
    ]
    
    results = {}
    
    for arch, name in architectures:
        print(f"\n=== Training {name} ===")
        
        mlp = MNISTMLP(hidden_layers=arch, learning_rate=0.001)
        mlp.build_model(input_shape=(28, 28), num_classes=10)
        
        # Train for fewer epochs for demonstration
        history = mlp.train(X_train, y_train, epochs=5, verbose=0)
        
        # Evaluate
        test_loss, test_accuracy, _ = mlp.evaluate(X_test, y_test)
        
        results[name] = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'model': mlp
        }
    
    # Compare results
    print("\n=== Architecture Comparison ===")
    for name, result in results.items():
        print(f"{name}: Test Accuracy = {result['test_accuracy']:.4f}")
    
    return results

def experiment_with_activations():
    """Experiment with different activation functions"""
    # Load data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    activations = ['relu', 'sigmoid', 'tanh']
    
    results = {}
    
    for activation in activations:
        print(f"\n=== Training with {activation} activation ===")
        
        mlp = MNISTMLP(hidden_layers=[128, 64], activation=activation, learning_rate=0.001)
        mlp.build_model(input_shape=(28, 28), num_classes=10)
        
        # Train for fewer epochs for demonstration
        history = mlp.train(X_train, y_train, epochs=5, verbose=0)
        
        # Evaluate
        test_loss, test_accuracy, _ = mlp.evaluate(X_test, y_test)
        
        results[activation] = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'model': mlp
        }
    
    # Compare results
    print("\n=== Activation Function Comparison ===")
    for activation, result in results.items():
        print(f"{activation}: Test Accuracy = {result['test_accuracy']:.4f}")
    
    return results

def demonstrate_overfitting():
    """Demonstrate overfitting and regularization techniques"""
    # Load data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Use smaller training set to encourage overfitting
    X_train_small = X_train[:1000]
    y_train_small = y_train[:1000]
    
    print("=== Demonstrating Overfitting ===")
    print(f"Using only {len(X_train_small)} training samples")
    
    # Model without regularization (likely to overfit)
    model_no_reg = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model_no_reg.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    
    # Model with regularization
    model_with_reg = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    model_with_reg.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
    
    # Train both models
    print("Training model without regularization...")
    history_no_reg = model_no_reg.fit(X_train_small, y_train_small,
                                     epochs=20, batch_size=32,
                                     validation_data=(X_test, y_test),
                                     verbose=0)
    
    print("Training model with regularization...")
    history_with_reg = model_with_reg.fit(X_train_small, y_train_small,
                                         epochs=20, batch_size=32,
                                         validation_data=(X_test, y_test),
                                         verbose=0)
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_no_reg.history['loss'], label='Training Loss (No Reg)')
    plt.plot(history_no_reg.history['val_loss'], label='Validation Loss (No Reg)')
    plt.plot(history_with_reg.history['loss'], '--', label='Training Loss (With Reg)')
    plt.plot(history_with_reg.history['val_loss'], '--', label='Validation Loss (With Reg)')
    plt.title('Loss: Regularization Effect')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_no_reg.history['accuracy'], label='Training Accuracy (No Reg)')
    plt.plot(history_no_reg.history['val_accuracy'], label='Validation Accuracy (No Reg)')
    plt.plot(history_with_reg.history['accuracy'], '--', label='Training Accuracy (With Reg)')
    plt.plot(history_with_reg.history['val_accuracy'], '--', label='Validation Accuracy (With Reg)')
    plt.title('Accuracy: Regularization Effect')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Final evaluation
    test_loss_no_reg, test_acc_no_reg = model_no_reg.evaluate(X_test, y_test, verbose=0)
    test_loss_with_reg, test_acc_with_reg = model_with_reg.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nFinal Test Results:")
    print(f"Without regularization: Loss = {test_loss_no_reg:.4f}, Accuracy = {test_acc_no_reg:.4f}")
    print(f"With regularization: Loss = {test_loss_with_reg:.4f}, Accuracy = {test_acc_with_reg:.4f}")
    print(f"Improvement with regularization: {test_acc_with_reg - test_acc_no_reg:.4f}")

def main():
    """Main function to run the complete MNIST MLP example"""
    print("=== MNIST Digit Classification with MLP ===")
    
    # Initialize MLP
    mlp = MNISTMLP(
        hidden_layers=[256, 128, 64],
        activation='relu',
        output_activation='softmax',
        learning_rate=0.001
    )
    
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = mlp.load_and_preprocess_data()
    
    # Build model
    mlp.build_model(input_shape=(28, 28), num_classes=10)
    
    # Display model architecture
    print("\nModel Architecture:")
    mlp.model.summary()
    
    # Train model
    history = mlp.train(X_train, y_train, epochs=15, batch_size=128, verbose=1)
    
    # Evaluate model
    test_loss, test_accuracy, y_pred = mlp.evaluate(X_test, y_test)
    
    # Plot training history
    mlp.plot_training_history()
    
    # Plot confusion matrix
    mlp.plot_confusion_matrix(y_test, y_pred)
    
    # Visualize some predictions
    mlp.visualize_predictions(X_test, y_test, num_samples=10)
    
    # Additional experiments
    print("\n" + "="*50)
    print("Running Additional Experiments...")
    
    # Experiment with architectures
    arch_results = experiment_with_architectures()
    
    # Experiment with activations
    activation_results = experiment_with_activations()
    
    # Demonstrate overfitting
    demonstrate_overfitting()
    
    return mlp, test_accuracy

if __name__ == "__main__":
    # Run the main MNIST classification
    trained_mlp, final_accuracy = main()
    
    print(f"\n=== Final Results ===")
    print(f"MNIST classification test accuracy: {final_accuracy:.4f}")
    print("MLP training and evaluation completed successfully!")