import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import pathlib
import warnings
warnings.filterwarnings('ignore')

class TransferLearningClassifier:
    """Transfer Learning for Cats vs Dogs classification"""
    
    def __init__(self, data_dir=None, img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = ['cat', 'dog']
        
    def download_and_prepare_data(self):
        """Download and prepare Cats vs Dogs dataset"""
        print("Preparing Cats vs Dogs dataset...")
        
        # Download dataset if not already available
        if self.data_dir is None:
            self.data_dir = pathlib.Path("cats_vs_dogs")
            if not self.data_dir.exists():
                print("Downloading dataset...")
                dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
                keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)
                self.data_dir = pathlib.Path("datasets/cats_and_dogs_filtered")
        
        # Define paths
        train_dir = self.data_dir / 'train'
        validation_dir = self.data_dir / 'validation'
        
        # Create datasets if they don't exist (for demonstration)
        if not train_dir.exists():
            self.create_sample_dataset()
            train_dir = self.data_dir / 'train'
            validation_dir = self.data_dir / 'validation'
        
        print(f"Training directory: {train_dir}")
        print(f"Validation directory: {validation_dir}")
        
        return train_dir, validation_dir
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        print("Creating sample dataset for demonstration...")
        
        # Create directories
        (self.data_dir / 'train' / 'cats').mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'train' / 'dogs').mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'validation' / 'cats').mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'validation' / 'dogs').mkdir(parents=True, exist_ok=True)
        
        # Note: In a real scenario, you would copy actual cat/dog images here
        # For this demo, we'll work with the actual dataset structure
        
    def create_data_generators(self, train_dir, validation_dir):
        """Create data generators with augmentation"""
        print("Creating data generators...")
        
        # Data augmentation for training
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        print(f"Classes: {train_generator.class_indices}")
        
        return train_generator, validation_generator
    
    def build_vgg16_model(self, trainable_layers=0):
        """Build model using VGG16 pre-trained weights"""
        print("Building VGG16-based model...")
        
        # Load pre-trained VGG16 model
        base_model = applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Make last few layers trainable if specified
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
        
        # Add custom classifier on top
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        return model, base_model
    
    def build_resnet50_model(self, trainable_layers=0):
        """Build model using ResNet50 pre-trained weights"""
        print("Building ResNet50-based model...")
        
        # Load pre-trained ResNet50 model
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Make last few layers trainable if specified
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
        
        # Add custom classifier on top
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model, base_model
    
    def build_mobilenet_model(self, trainable_layers=0):
        """Build model using MobileNetV2 pre-trained weights"""
        print("Building MobileNetV2-based model...")
        
        # Load pre-trained MobileNetV2 model
        base_model = applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Make last few layers trainable if specified
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
        
        # Add custom classifier on top
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model, base_model
    
    def compile_model(self, model, learning_rate=0.0001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model(self, model, train_generator, validation_generator, epochs=20):
        """Train the transfer learning model"""
        print("Training transfer learning model...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_transfer_learning_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, model, validation_generator):
        """Evaluate the model"""
        print("Evaluating model...")
        
        # Evaluate metrics
        results = model.evaluate(validation_generator, verbose=0)
        metrics = dict(zip(model.metrics_names, results))
        
        print("Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Make predictions
        validation_generator.reset()
        y_pred = model.predict(validation_generator, 
                              steps=validation_generator.samples // validation_generator.batch_size + 1)
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        
        # Get true labels
        y_true = validation_generator.classes[:len(y_pred_classes)]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, 
                                  target_names=self.class_names))
        
        return metrics, y_true, y_pred_classes, y_pred
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot precision
        ax3.plot(self.history.history['precision'], label='Training Precision')
        ax3.plot(self.history.history['val_precision'], label='Validation Precision')
        ax3.set_title('Model Precision')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot recall
        ax4.plot(self.history.history['recall'], label='Training Recall')
        ax4.plot(self.history.history['val_recall'], label='Validation Recall')
        ax4.set_title('Model Recall')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred_classes):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix - Cats vs Dogs Classification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    def visualize_predictions(self, model, validation_generator, num_samples=8):
        """Visualize predictions with confidence scores"""
        validation_generator.reset()
        images, true_labels = next(validation_generator)
        
        # Make predictions
        predictions = model.predict(images)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        confidence_scores = np.maximum(predictions, 1 - predictions).flatten()
        
        # Plot
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            axes[i].imshow(images[i])
            true_label = self.class_names[int(true_labels[i])]
            pred_label = self.class_names[predicted_classes[i]]
            confidence = confidence_scores[i]
            
            title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}'
            axes[i].set_title(title)
            axes[i].axis('off')
            
            # Color title based on correctness
            if true_label == pred_label:
                axes[i].title.set_color('green')
            else:
                axes[i].title.set_color('red')
        
        plt.tight_layout()
        plt.show()
    
    def fine_tune_model(self, model, base_model, train_generator, validation_generator, 
                       initial_epochs=10, fine_tune_epochs=10, learning_rate=0.00001):
        """Fine-tune the model by unfreezing some base layers"""
        print("Fine-tuning the model...")
        
        # Unfreeze the base model
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) // 2
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Continue training
        total_epochs = initial_epochs + fine_tune_epochs
        
        fine_tune_history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=total_epochs,
            initial_epoch=self.history.epoch[-1] + 1,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=1
        )
        
        # Combine histories
        for key in self.history.history.keys():
            self.history.history[key].extend(fine_tune_history.history[key])
        
        return self.history

def compare_pre_trained_models():
    """Compare different pre-trained models"""
    # Initialize classifier
    tl_classifier = TransferLearningClassifier()
    
    # Prepare data
    train_dir, validation_dir = tl_classifier.download_and_prepare_data()
    train_generator, validation_generator = tl_classifier.create_data_generators(
        train_dir, validation_dir)
    
    models_config = {
        'VGG16': tl_classifier.build_vgg16_model,
        'ResNet50': tl_classifier.build_resnet50_model,
        'MobileNetV2': tl_classifier.build_mobilenet_model
    }
    
    results = {}
    
    for model_name, model_builder in models_config.items():
        print(f"\n{'='*50}")
        print(f"Training with {model_name}")
        print(f"{'='*50}")
        
        # Build model
        model, base_model = model_builder(trainable_layers=0)
        
        # Compile model
        model = tl_classifier.compile_model(model, learning_rate=0.0001)
        
        # Display model summary
        print(f"\n{model_name} Architecture:")
        print(f"Trainable weights: {sum([w.shape.num_elements() for w in model.trainable_weights])}")
        print(f"Non-trainable weights: {sum([w.shape.num_elements() for w in model.non_trainable_weights])}")
        
        # Train model (fewer epochs for demonstration)
        history = tl_classifier.train_model(model, train_generator, validation_generator, epochs=10)
        
        # Evaluate model
        metrics, y_true, y_pred_classes, y_pred = tl_classifier.evaluate_model(
            model, validation_generator)
        
        results[model_name] = {
            'model': model,
            'base_model': base_model,
            'test_accuracy': metrics['accuracy'],
            'test_loss': metrics['loss'],
            'history': history
        }
    
    # Compare results
    print("\n" + "="*60)
    print("PRE-TRAINED MODEL COMPARISON RESULTS")
    print("="*60)
    for name, result in results.items():
        print(f"{name}: Test Accuracy = {result['test_accuracy']:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in names]
    
    bars = plt.bar(names, accuracies, color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Test Accuracy')
    plt.title('Pre-trained Model Comparison on Cats vs Dogs')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

def demonstrate_feature_extraction():
    """Demonstrate feature extraction without fine-tuning"""
    print("Demonstrating Feature Extraction approach...")
    
    tl_classifier = TransferLearningClassifier()
    train_dir, validation_dir = tl_classifier.download_and_prepare_data()
    train_generator, validation_generator = tl_classifier.create_data_generators(
        train_dir, validation_dir)
    
    # Build model with feature extraction (no fine-tuning)
    model, base_model = tl_classifier.build_vgg16_model(trainable_layers=0)
    model = tl_classifier.compile_model(model, learning_rate=0.0001)
    
    print("Feature Extraction Model Summary:")
    model.summary()
    
    # Train with feature extraction only
    history = tl_classifier.train_model(model, train_generator, validation_generator, epochs=10)
    
    # Evaluate
    metrics, y_true, y_pred_classes, y_pred = tl_classifier.evaluate_model(
        model, validation_generator)
    
    tl_classifier.plot_training_history()
    tl_classifier.plot_confusion_matrix(y_true, y_pred_classes)
    
    return model, metrics

def main():
    """Main function to run transfer learning for Cats vs Dogs"""
    print("=== Cats vs Dogs Classification with Transfer Learning ===")
    
    # Initialize transfer learning classifier
    tl_classifier = TransferLearningClassifier(batch_size=32, img_size=(224, 224))
    
    # Prepare data
    train_dir, validation_dir = tl_classifier.download_and_prepare_data()
    train_generator, validation_generator = tl_classifier.create_data_generators(
        train_dir, validation_dir)
    
    # Build VGG16-based model
    model, base_model = tl_classifier.build_vgg16_model(trainable_layers=0)
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Compile model
    model = tl_classifier.compile_model(model, learning_rate=0.0001)
    
    # Train model (feature extraction phase)
    print("\n=== Phase 1: Feature Extraction ===")
    history = tl_classifier.train_model(model, train_generator, validation_generator, epochs=15)
    
    # Evaluate model
    metrics, y_true, y_pred_classes, y_pred = tl_classifier.evaluate_model(
        model, validation_generator)
    
    # Plot results
    tl_classifier.plot_training_history()
    tl_classifier.plot_confusion_matrix(y_true, y_pred_classes)
    tl_classifier.visualize_predictions(model, validation_generator)
    
    # Fine-tuning phase
    print("\n=== Phase 2: Fine-Tuning ===")
    fine_tune_history = tl_classifier.fine_tune_model(
        model, base_model, train_generator, validation_generator,
        initial_epochs=15, fine_tune_epochs=10, learning_rate=0.00001
    )
    
    # Evaluate after fine-tuning
    print("\n=== After Fine-Tuning ===")
    metrics_ft, y_true_ft, y_pred_classes_ft, y_pred_ft = tl_classifier.evaluate_model(
        model, validation_generator)
    
    # Plot fine-tuning results
    tl_classifier.plot_training_history()
    tl_classifier.plot_confusion_matrix(y_true_ft, y_pred_classes_ft)
    
    # Additional experiments
    print("\n" + "="*60)
    print("Running Additional Experiments...")
    
    # Compare pre-trained models
    model_comparison = compare_pre_trained_models()
    
    # Demonstrate feature extraction
    feature_extraction_model, fe_metrics = demonstrate_feature_extraction()
    
    return tl_classifier, model, metrics_ft

if __name__ == "__main__":
    # Run main transfer learning classification
    tl_classifier, trained_model, final_metrics = main()
    
    print(f"\n=== Final Results ===")
    print(f"Cats vs Dogs classification accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print("Transfer learning training and evaluation completed successfully!")