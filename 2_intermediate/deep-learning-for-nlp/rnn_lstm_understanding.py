import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class RNNUnderstanding:
    """Class to demonstrate and understand RNNs and LSTMs"""
    
    def __init__(self):
        self.models = {}
    
    def demonstrate_sequence_processing(self):
        """Demonstrate sequence processing with simple examples"""
        print("=== Sequence Processing Demonstration ===")
        
        # Simple sequence example
        sequence = [1, 2, 3, 4, 5]
        print(f"Input sequence: {sequence}")
        
        # Manual RNN-like processing
        hidden_state = 0
        print("\nManual RNN Processing:")
        for i, x in enumerate(sequence):
            # Simple RNN computation: h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
            # Using simplified version for demonstration
            hidden_state = np.tanh(0.5 * hidden_state + 0.8 * x + 0.1)
            print(f"Step {i+1}: input={x}, hidden_state={hidden_state:.4f}")
        
        return sequence, hidden_state
    
    def create_simple_rnn_model(self, units=32, input_shape=(None, 1)):
        """Create a simple RNN model"""
        model = models.Sequential([
            layers.SimpleRNN(units, return_sequences=True, input_shape=input_shape),
            layers.SimpleRNN(units//2),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def create_lstm_model(self, units=32, input_shape=(None, 1)):
        """Create an LSTM model"""
        model = models.Sequential([
            layers.LSTM(units, return_sequences=True, input_shape=input_shape),
            layers.LSTM(units//2),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def create_gru_model(self, units=32, input_shape=(None, 1)):
        """Create a GRU model"""
        model = models.Sequential([
            layers.GRU(units, return_sequences=True, input_shape=input_shape),
            layers.GRU(units//2),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def visualize_rnn_cell(self):
        """Visualize RNN cell structure"""
        # Create a simple RNN layer to visualize
        rnn_layer = layers.SimpleRNN(3, return_sequences=True, return_state=True)
        
        # Create sample input
        sample_input = tf.constant([[[1.0], [2.0], [3.0]]])  # Shape: (1, 3, 1)
        
        # Get outputs
        whole_sequence_output, final_state = rnn_layer(sample_input)
        
        print("RNN Cell Demonstration:")
        print(f"Input shape: {sample_input.shape}")
        print(f"Whole sequence output shape: {whole_sequence_output.shape}")
        print(f"Final state shape: {final_state.shape}")
        print(f"Final state values: {final_state.numpy()}")
        
        # Plot RNN unrolling
        self._plot_rnn_unrolling()
        
        return whole_sequence_output, final_state
    
    def _plot_rnn_unrolling(self):
        """Plot RNN unrolling concept"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RNN cell
        ax1.text(0.5, 0.5, 'RNN Cell', ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax1.annotate('h_{t-1}', xy=(0.1, 0.5), xytext=(-0.2, 0.5),
                    arrowprops=dict(arrowstyle='->'), fontsize=12)
        ax1.annotate('x_t', xy=(0.5, 0.9), xytext=(0.5, 1.1),
                    arrowprops=dict(arrowstyle='->'), fontsize=12)
        ax1.annotate('h_t', xy=(0.9, 0.5), xytext=(1.2, 0.5),
                    arrowprops=dict(arrowstyle='->'), fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Single RNN Cell', fontsize=14)
        ax1.axis('off')
        
        # Unrolled RNN
        positions = [(0.1, 0.5), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.9, 0.5)]
        for i, (x, y) in enumerate(positions):
            ax2.text(x, y, f'RNN\nCell {i+1}', ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
            
            if i > 0:
                # Draw connection from previous cell
                ax2.annotate('', xy=(x-0.15, y), xytext=(positions[i-1][0]+0.15, y),
                           arrowprops=dict(arrowstyle='->'))
        
        # Inputs and outputs
        for i, (x, y) in enumerate(positions):
            ax2.text(x, y+0.3, f'x_{i+1}', ha='center', va='center', fontsize=8)
            ax2.text(x, y-0.3, f'h_{i+1}', ha='center', va='center', fontsize=8)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Unrolled RNN (Sequence Processing)', fontsize=14)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_lstm_gates(self):
        """Visualize LSTM gate mechanisms"""
        print("\n=== LSTM Gate Mechanisms ===")
        print("LSTM has three gates:")
        print("1. Forget Gate: Decides what information to throw away from cell state")
        print("2. Input Gate: Decides what new information to store in cell state")
        print("3. Output Gate: Decides what to output based on cell state")
        
        # Create LSTM layer to examine
        lstm_layer = layers.LSTM(4, return_sequences=True, return_state=True)
        
        # Sample input
        sample_input = tf.constant([[[1.0], [2.0], [3.0]]])
        
        # Get outputs
        whole_sequence_output, final_hidden_state, final_cell_state = lstm_layer(sample_input)
        
        print(f"\nLSTM Output Shapes:")
        print(f"Whole sequence output: {whole_sequence_output.shape}")
        print(f"Final hidden state: {final_hidden_state.shape}")
        print(f"Final cell state: {final_cell_state.shape}")
        
        # Plot LSTM cell structure
        self._plot_lstm_cell()
        
        return whole_sequence_output, final_hidden_state, final_cell_state
    
    def _plot_lstm_cell(self):
        """Plot LSTM cell structure"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # LSTM cell components
        components = {
            'Forget Gate': (0.3, 0.8),
            'Input Gate': (0.3, 0.5),
            'Output Gate': (0.3, 0.2),
            'Cell State': (0.7, 0.65),
            'Hidden State': (0.7, 0.35)
        }
        
        # Draw components
        for name, (x, y) in components.items():
            color = "lightcoral" if 'Gate' in name else "lightblue"
            ax.text(x, y, name, ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        
        # Draw connections
        # Forget gate to cell state
        ax.annotate('', xy=(0.5, 0.75), xytext=(0.4, 0.8),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        # Input gate to cell state
        ax.annotate('', xy=(0.5, 0.6), xytext=(0.4, 0.5),
                   arrowprops=dict(arrowstyle='->', color='green'))
        
        # Cell state to output gate
        ax.annotate('', xy=(0.6, 0.5), xytext=(0.6, 0.3),
                   arrowprops=dict(arrowstyle='->', color='blue'))
        
        # Input arrows
        ax.annotate('h_{t-1}', xy=(0.1, 0.65), xytext=(-0.1, 0.65),
                   arrowprops=dict(arrowstyle='->'), fontsize=10)
        ax.annotate('x_t', xy=(0.1, 0.35), xytext=(-0.1, 0.35),
                   arrowprops=dict(arrowstyle='->'), fontsize=10)
        
        # Output arrows
        ax.annotate('h_t', xy=(0.9, 0.35), xytext=(1.1, 0.35),
                   arrowprops=dict(arrowstyle='->'), fontsize=10)
        ax.annotate('C_t', xy=(0.9, 0.65), xytext=(1.1, 0.65),
                   arrowprops=dict(arrowstyle='->'), fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('LSTM Cell Architecture', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_vanishing_gradient(self):
        """Demonstrate the vanishing gradient problem"""
        print("\n=== Vanishing Gradient Problem ===")
        
        # Simulate gradient propagation through time
        time_steps = 10
        weights = np.random.randn(time_steps) * 0.5  # Small weights
        
        # Simulate gradient propagation
        gradient = 1.0  # Initial gradient
        gradients = [gradient]
        
        for t in range(time_steps):
            gradient = gradient * weights[t] * 0.5  # Simulate backpropagation
            gradients.append(gradient)
        
        print("Gradient propagation through time:")
        for t, grad in enumerate(gradients):
            print(f"Time step {t}: {grad:.6f}")
        
        # Plot gradient vanishing
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(gradients)), gradients, 'o-', linewidth=2, markersize=8)
        plt.axhline(y=0.01, color='r', linestyle='--', label='Vanishing threshold')
        plt.xlabel('Time Step')
        plt.ylabel('Gradient Magnitude')
        plt.title('Vanishing Gradient Problem')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()
        
        return gradients
    
    def compare_rnn_variants(self, sequence_length=10):
        """Compare different RNN variants on a simple task"""
        print("\n=== Comparing RNN Variants ===")
        
        # Create simple sequence prediction task
        # Input: sine wave, Output: next value prediction
        t = np.linspace(0, 4*np.pi, sequence_length + 1)
        X = np.sin(t[:-1]).reshape(1, sequence_length, 1)  # Input sequence
        y = np.sin(t[1:]).reshape(1, sequence_length, 1)   # Next values (for sequence output)
        
        print(f"Input sequence shape: {X.shape}")
        print(f"Target sequence shape: {y.shape}")
        
        # Define different RNN variants
        variants = {
            'Simple RNN': self.create_simple_rnn_model(units=16, input_shape=(sequence_length, 1)),
            'LSTM': self.create_lstm_model(units=16, input_shape=(sequence_length, 1)),
            'GRU': self.create_gru_model(units=16, input_shape=(sequence_length, 1))
        }
        
        results = {}
        
        for name, model in variants.items():
            print(f"\nTraining {name}...")
            
            # Compile model
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train on simple pattern
            history = model.fit(X, y, epochs=50, verbose=0)
            
            # Predict
            predictions = model.predict(X, verbose=0)
            mse = np.mean((predictions - y) ** 2)
            
            results[name] = {
                'model': model,
                'predictions': predictions,
                'mse': mse,
                'history': history
            }
            
            print(f"{name} - Final MSE: {mse:.6f}")
        
        # Plot predictions
        self._plot_rnn_comparisons(X, y, results)
        
        return results
    
    def _plot_rnn_comparisons(self, X, y, results):
        """Plot comparison of RNN variants"""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Predictions
        plt.subplot(1, 2, 1)
        plt.plot(X.flatten(), 'bo-', label='Input', alpha=0.7)
        plt.plot(y.flatten(), 'go-', label='True Output', alpha=0.7)
        
        for name, result in results.items():
            plt.plot(result['predictions'].flatten(), 'o-', label=f'{name} Prediction', alpha=0.7)
        
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('RNN Variants - Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Training loss
        plt.subplot(1, 2, 2)
        for name, result in results.items():
            plt.plot(result['history'].history['loss'], label=name)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('RNN Variants - Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_bidirectional_rnn(self):
        """Demonstrate bidirectional RNN"""
        print("\n=== Bidirectional RNN ===")
        print("Bidirectional RNN processes sequence in both directions:")
        print("- Forward pass: from start to end")
        print("- Backward pass: from end to start")
        print("- Concatenates both representations")
        
        # Create bidirectional layer
        bidirectional_layer = layers.Bidirectional(layers.LSTM(8, return_sequences=True))
        
        # Sample input
        sample_input = tf.constant([[[1.0], [2.0], [3.0], [4.0]]])  # Shape: (1, 4, 1)
        
        # Get output
        output = bidirectional_layer(sample_input)
        
        print(f"Input shape: {sample_input.shape}")
        print(f"Bidirectional LSTM output shape: {output.shape}")
        print("Note: Output features = 2 * units (forward + backward)")
        
        # Plot bidirectional concept
        self._plot_bidirectional_rnn()
        
        return output
    
    def _plot_bidirectional_rnn(self):
        """Plot bidirectional RNN concept"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Sequence
        sequence = ['x₁', 'x₂', 'x₃', 'x₄']
        positions = [(0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5)]
        
        # Draw sequence
        for i, (x, y) in enumerate(positions):
            ax.text(x, y, sequence[i], ha='center', va='center', fontsize=14,
                   bbox=dict(boxstyle="circle,pad=0.3", facecolor="lightgray"))
        
        # Forward arrows
        for i in range(len(positions) - 1):
            ax.annotate('', xy=positions[i+1], xytext=positions[i],
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        
        # Backward arrows
        for i in range(len(positions) - 1, 0, -1):
            ax.annotate('', xy=positions[i-1], xytext=positions[i],
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Labels
        ax.text(0.1, 0.7, 'Forward\nPass', ha='center', va='center', 
               fontsize=12, color='blue')
        ax.text(0.9, 0.7, 'Backward\nPass', ha='center', va='center', 
               fontsize=12, color='red')
        ax.text(0.5, 0.3, 'Concatenated\nOutput', ha='center', va='center', 
               fontsize=12, color='green')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Bidirectional RNN Concept', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate RNN and LSTM concepts"""
    print("=== Understanding RNNs and LSTMs ===")
    
    understanding = RNNUnderstanding()
    
    # 1. Demonstrate sequence processing
    print("\n1. Sequence Processing Basics")
    sequence, final_state = understanding.demonstrate_sequence_processing()
    
    # 2. Visualize RNN cells
    print("\n2. RNN Cell Structure")
    rnn_output, rnn_state = understanding.visualize_rnn_cell()
    
    # 3. Demonstrate vanishing gradient problem
    print("\n3. Vanishing Gradient Problem")
    gradients = understanding.demonstrate_vanishing_gradient()
    
    # 4. Visualize LSTM gates
    print("\n4. LSTM Gate Mechanisms")
    lstm_output, lstm_hidden, lstm_cell = understanding.visualize_lstm_gates()
    
    # 5. Compare RNN variants
    print("\n5. Comparing RNN Variants")
    comparison_results = understanding.compare_rnn_variants()
    
    # 6. Demonstrate bidirectional RNN
    print("\n6. Bidirectional RNN")
    bidirectional_output = understanding.demonstrate_bidirectional_rnn()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Key Concepts Covered:")
    print("✓ Sequence processing with RNNs")
    print("✓ RNN cell structure and unrolling")
    print("✓ Vanishing gradient problem")
    print("✓ LSTM gate mechanisms (forget, input, output)")
    print("✓ Comparison of RNN variants (Simple RNN, LSTM, GRU)")
    print("✓ Bidirectional RNN processing")
    
    return understanding, comparison_results

if __name__ == "__main__":
    understanding, results = main()
    
    print("\n=== RNN/LSTM Understanding Completed ===")
    print("All concepts demonstrated successfully!")