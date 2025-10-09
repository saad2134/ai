import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalysisLSTM:
    """LSTM-based Sentiment Analysis Model"""
    
    def __init__(self, max_features=20000, max_length=200, embedding_dim=100):
        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.history = None
        self.tokenizer = None
    
    def load_imdb_data(self, num_words=None):
        """Load IMDB movie reviews dataset"""
        if num_words is None:
            num_words = self.max_features
            
        print("Loading IMDB dataset...")
        (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(
            num_words=num_words
        )
        
        # Get word index
        word_index = keras.datasets.imdb.get_word_index()
        
        # Reverse word index to get words from indices
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        print(f"Training data: {X_train.shape[0]} samples")
        print(f"Test data: {X_test.shape[0]} samples")
        print(f"Vocabulary size: {num_words}")
        
        return (X_train, y_train), (X_test, y_test), reverse_word_index
    
    def decode_review(self, indices, reverse_word_index):
        """Decode review from indices to text"""
        # Adjust indices since word_index starts from 1
        reverse_word_index = {key: value for key, value in reverse_word_index.items()}
        reverse_word_index[0] = '<PAD>'
        reverse_word_index[1] = '<START>'
        reverse_word_index[2] = '<UNK>'
        reverse_word_index[3] = '<UNUSED>'
        
        return ' '.join([reverse_word_index.get(i, '?') for i in indices])
    
    def preprocess_data(self, X_train, X_test):
        """Pad sequences to ensure uniform length"""
        X_train = keras.preprocessing.sequence.pad_sequences(
            X_train, maxlen=self.max_length, padding='post', truncating='post'
        )
        X_test = keras.preprocessing.sequence.pad_sequences(
            X_test, maxlen=self.max_length, padding='post', truncating='post'
        )
        
        print(f"Padded training sequences shape: {X_train.shape}")
        print(f"Padded test sequences shape: {X_test.shape}")
        
        return X_train, X_test
    
    def build_simple_lstm_model(self):
        """Build a simple LSTM model"""
        print("Building Simple LSTM Model...")
        
        model = models.Sequential([
            layers.Embedding(self.max_features, self.embedding_dim, input_length=self.max_length),
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def build_bidirectional_lstm_model(self):
        """Build a Bidirectional LSTM model"""
        print("Building Bidirectional LSTM Model...")
        
        model = models.Sequential([
            layers.Embedding(self.max_features, self.embedding_dim, input_length=self.max_length),
            layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def build_stacked_lstm_model(self):
        """Build a Stacked LSTM model"""
        print("Building Stacked LSTM Model...")
        
        model = models.Sequential([
            layers.Embedding(self.max_features, self.embedding_dim, input_length=self.max_length),
            layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def build_glove_model(self, embedding_matrix):
        """Build model with pre-trained GloVe embeddings"""
        print("Building Model with GloVe Embeddings...")
        
        model = models.Sequential([
            layers.Embedding(
                input_dim=embedding_matrix.shape[0],
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                input_length=self.max_length,
                trainable=False  # Freeze embeddings
            ),
            layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Model compiled successfully!")
        return model
    
    def train_model(self, model, X_train, y_train, X_val=None, y_val=None, 
                   epochs=10, batch_size=128):
        """Train the LSTM model"""
        print("Training LSTM model...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_lstm_model.h5',
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        if X_val is not None:
            self.history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Create validation split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            self.history = model.fit(
                X_train_split, y_train_split,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val_split, y_val_split),
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model on test data"""
        print("Evaluating model on test data...")
        
        # Evaluate metrics
        results = model.evaluate(X_test, y_test, verbose=0)
        metrics = dict(zip(model.metrics_names, results))
        
        print("Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        return metrics, y_pred, y_pred_proba
    
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
        if 'precision' in self.history.history:
            ax3.plot(self.history.history['precision'], label='Training Precision')
            ax3.plot(self.history.history['val_precision'], label='Validation Precision')
            ax3.set_title('Model Precision')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Precision')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot recall
        if 'recall' in self.history.history:
            ax4.plot(self.history.history['recall'], label='Training Recall')
            ax4.plot(self.history.history['val_recall'], label='Validation Recall')
            ax4.set_title('Model Recall')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Recall')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix - Sentiment Analysis')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    def analyze_predictions(self, X_test, y_test, y_pred_proba, reverse_word_index, num_samples=10):
        """Analyze model predictions with sample reviews"""
        # Get indices of correct and incorrect predictions
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        correct_indices = np.where(y_pred == y_test)[0]
        incorrect_indices = np.where(y_pred != y_test)[0]
        
        print(f"Correct predictions: {len(correct_indices)}/{len(y_test)} ({len(correct_indices)/len(y_test):.2%})")
        print(f"Incorrect predictions: {len(incorrect_indices)}/{len(y_test)} ({len(incorrect_indices)/len(y_test):.2%})")
        
        # Display sample correct predictions
        print("\n" + "="*60)
        print("SAMPLE CORRECT PREDICTIONS")
        print("="*60)
        
        correct_samples = np.random.choice(correct_indices, min(5, len(correct_indices)), replace=False)
        for idx in correct_samples:
            review = self.decode_review(X_test[idx], reverse_word_index)
            sentiment = "Positive" if y_test[idx] == 1 else "Negative"
            confidence = y_pred_proba[idx][0] if y_test[idx] == 1 else 1 - y_pred_proba[idx][0]
            
            print(f"\nTrue: {sentiment} | Pred: {sentiment} | Confidence: {confidence:.4f}")
            print(f"Review: {review[:200]}...")
        
        # Display sample incorrect predictions
        if len(incorrect_indices) > 0:
            print("\n" + "="*60)
            print("SAMPLE INCORRECT PREDICTIONS")
            print("="*60)
            
            incorrect_samples = np.random.choice(incorrect_indices, min(5, len(incorrect_indices)), replace=False)
            for idx in incorrect_samples:
                review = self.decode_review(X_test[idx], reverse_word_index)
                true_sentiment = "Positive" if y_test[idx] == 1 else "Negative"
                pred_sentiment = "Positive" if y_pred[idx] == 1 else "Negative"
                confidence = y_pred_proba[idx][0]
                
                print(f"\nTrue: {true_sentiment} | Pred: {pred_sentiment} | Confidence: {confidence:.4f}")
                print(f"Review: {review[:200]}...")

def load_glove_embeddings(embedding_path, word_index, embedding_dim=100):
    """Load pre-trained GloVe embeddings"""
    print("Loading GloVe embeddings...")
    
    embeddings_index = {}
    try:
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print(f"GloVe file not found at {embedding_path}")
        print("Please download GloVe embeddings from: https://nlp.stanford.edu/projects/glove/")
        print("Using random embeddings instead.")
        return None
    
    print(f'Found {len(embeddings_index)} word vectors.')
    
    # Prepare embedding matrix
    num_words = min(len(word_index) + 1, 20000)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    # Calculate coverage
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    coverage = nonzero_elements / num_words
    print(f'Embedding coverage: {coverage:.2%}')
    
    return embedding_matrix

def compare_lstm_architectures():
    """Compare different LSTM architectures"""
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalysisLSTM(max_features=10000, max_length=200)
    
    # Load data
    (X_train, y_train), (X_test, y_test), reverse_word_index = sentiment_analyzer.load_imdb_data(10000)
    X_train, X_test = sentiment_analyzer.preprocess_data(X_train, X_test)
    
    # Create validation set
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    architectures = {
        'Simple LSTM': sentiment_analyzer.build_simple_lstm_model(),
        'Bidirectional LSTM': sentiment_analyzer.build_bidirectional_lstm_model(),
        'Stacked LSTM': sentiment_analyzer.build_stacked_lstm_model()
    }
    
    results = {}
    
    for name, model in architectures.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")
        
        # Compile model
        model = sentiment_analyzer.compile_model(model, learning_rate=0.001)
        
        # Display model summary
        print(f"\n{name} Architecture:")
        model.summary()
        
        # Train model (fewer epochs for demonstration)
        history = sentiment_analyzer.train_model(
            model, X_train_split, y_train_split, X_val_split, y_val_split, epochs=5
        )
        
        # Evaluate model
        metrics, y_pred, y_pred_proba = sentiment_analyzer.evaluate_model(model, X_test, y_test)
        
        results[name] = {
            'model': model,
            'test_accuracy': metrics['accuracy'],
            'test_loss': metrics['loss'],
            'history': history
        }
    
    # Compare results
    print("\n" + "="*60)
    print("LSTM ARCHITECTURE COMPARISON RESULTS")
    print("="*60)
    for name, result in results.items():
        print(f"{name}: Test Accuracy = {result['test_accuracy']:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in names]
    
    bars = plt.bar(names, accuracies, color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Test Accuracy')
    plt.title('LSTM Architecture Comparison on IMDB Reviews')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

def demonstrate_with_glove_embeddings():
    """Demonstrate using pre-trained GloVe embeddings"""
    sentiment_analyzer = SentimentAnalysisLSTM(max_features=10000, max_length=200)
    
    # Load data
    (X_train, y_train), (X_test, y_test), reverse_word_index = sentiment_analyzer.load_imdb_data(10000)
    X_train, X_test = sentiment_analyzer.preprocess_data(X_train, X_test)
    
    # Get word index (adjust for IMDB dataset)
    word_index = keras.datasets.imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    
    # Load GloVe embeddings
    embedding_matrix = load_glove_embeddings(
        'glove.6B.100d.txt',  # Update this path to your GloVe file
        word_index,
        embedding_dim=100
    )
    
    if embedding_matrix is not None:
        # Build model with GloVe embeddings
        model = sentiment_analyzer.build_glove_model(embedding_matrix)
        model = sentiment_analyzer.compile_model(model, learning_rate=0.001)
        
        # Train and evaluate
        history = sentiment_analyzer.train_model(model, X_train, y_train, epochs=5)
        metrics, y_pred, y_pred_proba = sentiment_analyzer.evaluate_model(model, X_test, y_test)
        
        sentiment_analyzer.plot_training_history()
        sentiment_analyzer.plot_confusion_matrix(y_test, y_pred)
        
        return model, metrics
    else:
        print("GloVe embeddings not available. Using random embeddings.")
        return None, None

def main():
    """Main function to run LSTM sentiment analysis"""
    print("=== LSTM Sentiment Analysis on IMDB Reviews ===")
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalysisLSTM(max_features=10000, max_length=200)
    
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test), reverse_word_index = sentiment_analyzer.load_imdb_data(10000)
    X_train, X_test = sentiment_analyzer.preprocess_data(X_train, X_test)
    
    # Build model (using Bidirectional LSTM for main demonstration)
    model = sentiment_analyzer.build_bidirectional_lstm_model()
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Compile model
    model = sentiment_analyzer.compile_model(model, learning_rate=0.001)
    
    # Create validation set
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train model
    history = sentiment_analyzer.train_model(
        model, X_train_split, y_train_split, X_val_split, y_val_split, epochs=10
    )
    
    # Evaluate model
    metrics, y_pred, y_pred_proba = sentiment_analyzer.evaluate_model(model, X_test, y_test)
    
    # Plot results
    sentiment_analyzer.plot_training_history()
    sentiment_analyzer.plot_confusion_matrix(y_test, y_pred)
    
    # Analyze predictions
    sentiment_analyzer.analyze_predictions(X_test, y_test, y_pred_proba, reverse_word_index)
    
    # Additional experiments
    print("\n" + "="*60)
    print("Running Additional Experiments...")
    
    # Compare LSTM architectures
    architecture_results = compare_lstm_architectures()
    
    # Demonstrate GloVe embeddings (if available)
    print("\n" + "="*60)
    print("GloVe Embeddings Demonstration")
    print("="*60)
    glove_model, glove_metrics = demonstrate_with_glove_embeddings()
    
    return sentiment_analyzer, model, metrics

if __name__ == "__main__":
    # Run main sentiment analysis
    analyzer, trained_model, final_metrics = main()
    
    print(f"\n=== Final Results ===")
    print(f"Sentiment analysis test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print("LSTM sentiment analysis completed successfully!")