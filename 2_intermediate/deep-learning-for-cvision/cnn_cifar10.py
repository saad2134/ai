import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import datetime

class CIFAR10CNN:
    """CNN for CIFAR-10 classification"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
    
    def load_and_preprocess_data(self):
        """Load and preprocess CIFAR-10 dataset"""
        print("Loading CIFAR-10 dataset...")
        
        # Load CIFAR-10 dataset
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        # Normalize pixel values to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Convert labels to categorical one-hot encoding
        y_train_categorical = keras.utils.to_categorical(y_train, 10)
        y_test_categorical = keras.utils.to_categorical(y_test, 10)
        
        return (X_train, y_train, y_train_categorical), (X_test, y_test, y_test_categorical)
    
    def build_simple_cnn(self):
        """Build a simple CNN architecture"""
        print("Building Simple CNN...")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def build_lenet_style(self):
        """Build a LeNet-style architecture adapted for CIFAR-10"""
        print("Building LeNet-style CNN...")
        
        model = models.Sequential([
            # First Convolutional Block (LeNet style)
            layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(16, (5, 5), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Modern additions for better performance
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(120, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(84, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def build_alexnet_style(self):
        """Build an AlexNet-style architecture adapted for CIFAR-10"""
        print("Building AlexNet-style CNN...")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(96, (3, 3), activation='relu', padding='same', 
                         input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """Compile the model with appropriate settings"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, X_train, y_train, X_val=None, y_val=None, 
                   epochs=50, batch_size=128):
        """Train the CNN model"""
        print("Training CNN model...")
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_cifar10_model.h5',
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        datagen.fit(X_train)
        
        # Train the model
        if X_val is not None:
            self.history = model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Create validation split if not provided
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            self.history = model.fit(
                datagen.flow(X_train_split, y_train_split, batch_size=batch_size),
                steps_per_epoch=len(X_train_split) // batch_size,
                epochs=epochs,
                validation_data=(X_val_split, y_val_split),
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history
    
    def evaluate_model(self, model, X_test, y_test, y_test_categorical):
        """Evaluate the model on test data"""
        print("Evaluating model on test data...")
        
        # Evaluate metrics
        test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                  target_names=self.class_names))
        
        return test_loss, test_accuracy, y_pred_classes
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
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
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix - CIFAR-10 Classification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self, X_test, y_test, y_pred_classes, num_samples=15):
        """Visualize test predictions"""
        # Select random samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(3, 5, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            axes[i].imshow(X_test[idx])
            true_label = self.class_names[y_test[idx][0]]
            pred_label = self.class_names[y_pred_classes[idx]]
            
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}')
            axes[i].axis('off')
            
            # Color title based on correctness
            if true_label == pred_label:
                axes[i].title.set_color('green')
            else:
                axes[i].title.set_color('red')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_feature_maps(self, model, X_sample, layer_name='conv2d_3'):
        """Visualize feature maps from a convolutional layer"""
        # Create a model that outputs the feature maps
        layer_output = model.get_layer(layer_name).output
        feature_map_model = models.Model(inputs=model.input, outputs=layer_output)
        
        # Get feature maps for a sample image
        feature_maps = feature_map_model.predict(X_sample[np.newaxis, ...])
        
        # Plot feature maps
        fig, axes = plt.subplots(4, 8, figsize=(15, 8))
        for i in range(min(32, feature_maps.shape[-1])):
            ax = axes[i//8, i%8]
            ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'FM {i+1}')
        
        plt.suptitle(f'Feature Maps from {layer_name}', fontsize=16)
        plt.tight_layout()
        plt.show()

def compare_cnn_architectures():
    """Compare different CNN architectures on CIFAR-10"""
    # Load data
    cnn = CIFAR10CNN()
    (X_train, y_train, y_train_categorical), (X_test, y_test, y_test_categorical) = cnn.load_and_preprocess_data()
    
    # Create validation set
    X_train_split, X_val_split, y_train_split, y_train_categorical_split, y_val_split, y_val_categorical_split = train_test_split(
        X_train, y_train, y_train_categorical, test_size=0.2, random_state=42
    )
    
    architectures = {
        'Simple CNN': cnn.build_simple_cnn(),
        'LeNet Style': cnn.build_lenet_style(),
        'AlexNet Style': cnn.build_alexnet_style()
    }
    
    results = {}
    
    for name, model in architectures.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")
        
        # Compile model
        model = cnn.compile_model(model, learning_rate=0.001)
        
        # Train model (fewer epochs for demonstration)
        history = cnn.train_model(model, X_train_split, y_train_categorical_split, 
                                X_val_split, y_val_categorical_split, epochs=20)
        
        # Evaluate model
        test_loss, test_accuracy, y_pred = cnn.evaluate_model(
            model, X_test, y_test, y_test_categorical)
        
        results[name] = {
            'model': model,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'history': history
        }
    
    # Compare results
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON RESULTS")
    print("="*60)
    for name, result in results.items():
        print(f"{name}: Test Accuracy = {result['test_accuracy']:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in names]
    
    bars = plt.bar(names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Test Accuracy')
    plt.title('CNN Architecture Comparison on CIFAR-10')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.4f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return results

def demonstrate_data_augmentation():
    """Demonstrate the effect of data augmentation"""
    # Load a sample image
    (X_train, y_train, _), _ = CIFAR10CNN().load_and_preprocess_data()
    
    # Create data augmentation generator
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    # Select a sample image
    sample_image = X_train[0]
    
    # Generate augmented images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    axes[0].imshow(sample_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    for i in range(1, 10):
        augmented = datagen.random_transform(sample_image)
        axes[i].imshow(augmented)
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run CIFAR-10 CNN classification"""
    print("=== CIFAR-10 Image Classification with CNN ===")
    
    # Initialize CNN
    cnn = CIFAR10CNN()
    
    # Load and preprocess data
    (X_train, y_train, y_train_categorical), (X_test, y_test, y_test_categorical) = cnn.load_and_preprocess_data()
    
    # Build model (using simple CNN for main demonstration)
    model = cnn.build_simple_cnn()
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Compile model
    model = cnn.compile_model(model, learning_rate=0.001)
    
    # Create validation set
    X_train_split, X_val_split, y_train_split, y_train_categorical_split, y_val_split, y_val_categorical_split = train_test_split(
        X_train, y_train, y_train_categorical, test_size=0.2, random_state=42
    )
    
    # Train model
    history = cnn.train_model(model, X_train_split, y_train_categorical_split, 
                            X_val_split, y_val_categorical_split, epochs=50)
    
    # Evaluate model
    test_loss, test_accuracy, y_pred_classes = cnn.evaluate_model(
        model, X_test, y_test, y_test_categorical)
    
    # Plot training history
    cnn.plot_training_history()
    
    # Plot confusion matrix
    cnn.plot_confusion_matrix(y_test, y_pred_classes)
    
    # Visualize predictions
    cnn.visualize_predictions(X_test, y_test, y_pred_classes)
    
    # Visualize feature maps for a sample image
    sample_idx = np.random.randint(0, len(X_test))
    cnn.visualize_feature_maps(model, X_test[sample_idx])
    
    # Additional experiments
    print("\n" + "="*60)
    print("Running Additional Experiments...")
    
    # Compare architectures
    architecture_results = compare_cnn_architectures()
    
    # Demonstrate data augmentation
    demonstrate_data_augmentation()
    
    return cnn, model, test_accuracy

if __name__ == "__main__":
    # Run main CIFAR-10 classification
    cnn, trained_model, final_accuracy = main()
    
    print(f"\n=== Final Results ===")
    print(f"CIFAR-10 classification test accuracy: {final_accuracy:.4f}")
    print("CNN training and evaluation completed successfully!")