import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class PyTorchMLP(nn.Module):
    """MLP for MNIST classification using PyTorch"""
    
    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], num_classes=10, 
                 activation='relu', dropout_rate=0.3):
        super(PyTorchMLP, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        # Build hidden layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Add activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            # Add batch normalization
            layers.append(nn.BatchNorm1d(hidden_size))
            
            # Add dropout
            layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten the input if it's 2D (images)
        x = x.view(x.size(0), -1)
        return self.network(x)

class MNISTTrainer:
    """Trainer class for PyTorch MLP"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def load_data(self, batch_size=128):
        """Load and preprocess MNIST data"""
        print("Loading MNIST dataset...")
        
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100. * correct / total
        
        return val_loss, val_accuracy
    
    def train(self, train_loader, val_loader, epochs=10, learning_rate=0.001):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        print("Starting training...")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_accuracy = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        return self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies
    
    def evaluate(self, test_loader):
        """Evaluate the model on test data"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        test_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / total
        
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        
        return test_loss, test_accuracy, np.array(all_predictions), np.array(all_targets)
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def visualize_predictions(model, test_loader, device, num_samples=10):
    """Visualize some test predictions"""
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Move to device
    images, labels = images.to(device), labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Convert to numpy for plotting
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'True: {labels[i]}, Pred: {predicted[i]}')
        axes[i].axis('off')
        
        # Color title based on correctness
        if labels[i] == predicted[i]:
            axes[i].title.set_color('green')
        else:
            axes[i].title.set_color('red')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def experiment_with_different_models():
    """Experiment with different MLP architectures in PyTorch"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    architectures = [
        ([128], "Small MLP (128)"),
        ([256, 128], "Medium MLP (256-128)"),
        ([512, 256, 128], "Large MLP (512-256-128)"),
        ([256, 256, 256], "Wide MLP (256-256-256)")
    ]
    
    results = {}
    
    for hidden_sizes, name in architectures:
        print(f"\n=== Training {name} ===")
        
        # Create model
        model = PyTorchMLP(hidden_sizes=hidden_sizes, dropout_rate=0.3)
        trainer = MNISTTrainer(model, device)
        
        # Load data
        train_loader, test_loader = trainer.load_data(batch_size=128)
        
        # Split training data for validation
        train_size = int(0.8 * len(train_loader.dataset))
        val_size = len(train_loader.dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_loader.dataset, [train_size, val_size])
        
        train_loader_sub = torch.utils.data.DataLoader(
            train_subset, batch_size=128, shuffle=True)
        val_loader_sub = torch.utils.data.DataLoader(
            val_subset, batch_size=128, shuffle=False)
        
        # Train for fewer epochs for demonstration
        trainer.train(train_loader_sub, val_loader_sub, epochs=5, learning_rate=0.001)
        
        # Evaluate on test set
        test_loss, test_accuracy, _, _ = trainer.evaluate(test_loader)
        
        results[name] = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'trainer': trainer
        }
    
    # Compare results
    print("\n=== Architecture Comparison ===")
    for name, result in results.items():
        print(f"{name}: Test Accuracy = {result['test_accuracy']:.2f}%")
    
    return results

def main():
    """Main function to run PyTorch MLP on MNIST"""
    print("=== MNIST Classification with PyTorch MLP ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = PyTorchMLP(
        input_size=784,
        hidden_sizes=[256, 128, 64],
        num_classes=10,
        activation='relu',
        dropout_rate=0.3
    )
    
    print("Model Architecture:")
    print(model)
    
    # Create trainer
    trainer = MNISTTrainer(model, device)
    
    # Load data
    train_loader, test_loader = trainer.load_data(batch_size=128)
    
    # Split training data for validation
    train_size = int(0.8 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size])
    
    train_loader_sub = torch.utils.data.DataLoader(
        train_subset, batch_size=128, shuffle=True)
    val_loader_sub = torch.utils.data.DataLoader(
        val_subset, batch_size=128, shuffle=False)
    
    # Train model
    train_losses, train_accs, val_losses, val_accs = trainer.train(
        train_loader_sub, val_loader_sub, epochs=10, learning_rate=0.001)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on test set
    test_loss, test_accuracy, y_pred, y_true = trainer.evaluate(test_loader)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, device, num_samples=10)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Additional experiments
    print("\n" + "="*50)
    print("Running Architecture Experiments...")
    arch_results = experiment_with_different_models()
    
    return trainer, test_accuracy

if __name__ == "__main__":
    # Run PyTorch MLP
    trainer, final_accuracy = main()
    
    print(f"\n=== Final Results ===")
    print(f"PyTorch MLP Test Accuracy: {final_accuracy:.2f}%")
    print("PyTorch MLP training and evaluation completed successfully!")