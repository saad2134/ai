import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def create_non_linear_datasets():
    """Create non-linearly separable datasets"""
    # Moons dataset
    X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=42)
    
    # Circles dataset
    X_circles, y_circles = make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=42)
    
    # XOR dataset
    np.random.seed(42)
    X_xor = np.random.randn(300, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0).astype(int)
    
    datasets = {
        'Moons': (X_moons, y_moons),
        'Circles': (X_circles, y_circles),
        'XOR': (X_xor, y_xor)
    }
    
    return datasets

def compare_models(X, y, dataset_name):
    """Compare SVM with kernel vs Logistic Regression"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM Linear': SVC(kernel='linear', random_state=42),
        'SVM RBF': SVC(kernel='rbf', random_state=42),
        'SVM Polynomial': SVC(kernel='poly', degree=3, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    # Print results
    print(f"\n=== {dataset_name} Dataset ===")
    for name, result in results.items():
        print(f"{name}: Accuracy = {result['accuracy']:.4f}")
    
    # Plot decision boundaries
    plot_decision_boundaries(X, y, models, dataset_name)
    
    return results

def plot_decision_boundaries(X, y, models, dataset_name):
    """Plot decision boundaries for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Scale the mesh grid
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    
    for idx, (name, model_info) in enumerate(models.items()):
        model = model_info['model']
        
        # Predict on mesh grid
        Z = model.predict(mesh_points_scaled)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        axes[idx].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
        axes[idx].set_title(f'{name}\nAccuracy: {model_info["accuracy"]:.4f}')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.suptitle(f'Decision Boundaries - {dataset_name} Dataset', y=1.02, fontsize=16)
    plt.show()

def svm_hyperparameter_tuning():
    """Demonstrate SVM hyperparameter tuning"""
    from sklearn.model_selection import GridSearchCV
    
    # Create complex dataset
    X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1, 10],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # Grid search
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print("=== SVM Hyperparameter Tuning Results ===")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Test best model
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return best_svm, grid_search.cv_results_

# Run the comparisons
if __name__ == "__main__":
    datasets = create_non_linear_datasets()
    
    for dataset_name, (X, y) in datasets.items():
        results = compare_models(X, y, dataset_name)
    
    # Demonstrate hyperparameter tuning
    best_model, cv_results = svm_hyperparameter_tuning()