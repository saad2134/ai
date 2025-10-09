import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import datetime
import warnings
warnings.filterwarnings('ignore')

class MLOpsPipeline:
    """Basic MLOps pipeline with model tracking and monitoring"""
    
    def __init__(self):
        self.model = None
        self.model_version = "1.0"
        self.metadata = {}
        self.performance_history = []
    
    def generate_sample_data(self, n_samples=1000):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        data = {
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(5, 2, n_samples),
            'feature3': np.random.randint(0, 10, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def train_model(self, X_train, y_train):
        """Train a model with tracking"""
        print("Training model...")
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Store training metadata
        self.metadata = {
            'model_type': 'RandomForest',
            'version': self.model_version,
            'training_date': datetime.datetime.now().isoformat(),
            'features_used': X_train.columns.tolist(),
            'training_samples': len(X_train),
            'parameters': {
                'n_estimators': 100,
                'random_state': 42
            }
        }
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model and track performance"""
        if self.model is None:
            print("No model trained yet.")
            return None
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store performance metrics
        performance = {
            'timestamp': datetime.datetime.now().isoformat(),
            'accuracy': accuracy,
            'test_samples': len(X_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.performance_history.append(performance)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        return performance
    
    def save_model_artifacts(self, base_path='model_artifacts'):
        """Save model and all artifacts"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save model
        model_path = f"{base_path}/model_v{self.model_version}.joblib"
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata_path = f"{base_path}/metadata_v{self.model_version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save performance history
        performance_path = f"{base_path}/performance_v{self.model_version}.json"
        with open(performance_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        print(f"Model artifacts saved to {base_path}/")
    
    def load_model_artifacts(self, base_path='model_artifacts', version='1.0'):
        """Load model and artifacts"""
        try:
            model_path = f"{base_path}/model_v{version}.joblib"
            self.model = joblib.load(model_path)
            
            metadata_path = f"{base_path}/metadata_v{version}.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            performance_path = f"{base_path}/performance_v{version}.json"
            with open(performance_path, 'r') as f:
                self.performance_history = json.load(f)
            
            print(f"Model v{version} loaded successfully")
            return True
        except FileNotFoundError:
            print("Model artifacts not found")
            return False
    
    def model_monitoring(self, X_new, y_new=None):
        """Monitor model performance on new data"""
        if self.model is None:
            print("No model loaded.")
            return None
        
        # Make predictions
        predictions = self.model.predict(X_new)
        
        # Calculate performance if true labels are available
        if y_new is not None:
            accuracy = accuracy_score(y_new, predictions)
            print(f"Current accuracy on new data: {accuracy:.4f}")
            
            # Compare with historical performance
            if self.performance_history:
                historical_acc = self.performance_history[-1]['accuracy']
                accuracy_change = accuracy - historical_acc
                print(f"Accuracy change from last evaluation: {accuracy_change:+.4f}")
                
                # Simple drift detection
                if abs(accuracy_change) > 0.05:  # 5% threshold
                    print("⚠️  Significant performance drift detected!")
        
        return predictions
    
    def retrain_model(self, new_data, target_col='target'):
        """Retrain model on new data"""
        print("Retraining model on new data...")
        
        X_new = new_data.drop(target_col, axis=1)
        y_new = new_data[target_col]
        
        # Retrain model
        self.model_version = f"{float(self.model_version) + 0.1:.1f}"
        self.train_model(X_new, y_new)
        
        print(f"Model retrained. New version: {self.model_version}")

def demonstrate_mlops_pipeline():
    """Demonstrate complete MLOps pipeline"""
    print("=== MLOps Pipeline Demonstration ===\n")
    
    mlops = MLOpsPipeline()
    
    # Generate initial data
    print("1. Generating initial dataset...")
    data_v1 = mlops.generate_sample_data(1000)
    X = data_v1.drop('target', axis=1)
    y = data_v1['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train initial model
    print("\n2. Training initial model...")
    mlops.train_model(X_train, y_train)
    
    # Evaluate model
    print("\n3. Evaluating model...")
    mlops.evaluate_model(X_test, y_test)
    
    # Save artifacts
    print("\n4. Saving model artifacts...")
    mlops.save_model_artifacts()
    
    # Simulate new data (concept drift)
    print("\n5. Simulating new data with concept drift...")
    data_v2 = mlops.generate_sample_data(500)
    # Introduce some drift by shifting feature distributions
    data_v2['feature1'] = data_v2['feature1'] + 1  # Shift distribution
    data_v2['feature2'] = data_v2['feature2'] * 1.5  # Scale distribution
    
    # Monitor performance on new data
    print("\n6. Monitoring model on new data...")
    X_new = data_v2.drop('target', axis=1)
    y_new = data_v2['target']
    mlops.model_monitoring(X_new, y_new)
    
    # Retrain model
    print("\n7. Retraining model on new data...")
    mlops.retrain_model(data_v2)
    mlops.evaluate_model(X_new, y_new)
    
    # Save new version
    print("\n8. Saving new model version...")
    mlops.save_model_artifacts()
    
    # Load and verify
    print("\n9. Loading original model...")
    mlops_v1 = MLOpsPipeline()
    mlops_v1.load_model_artifacts(version='1.0')
    
    print("\n10. Loading new model...")
    mlops_v2 = MLOpsPipeline()
    mlops_v2.load_model_artifacts(version='1.1')
    
    return mlops_v1, mlops_v2

def create_mlops_checklist():
    """Create MLOps best practices checklist"""
    checklist = {
        'Data Management': [
            '✓ Data versioning implemented',
            '✓ Data validation checks in place', 
            '✓ Feature store considered',
            '✓ Data pipeline monitoring'
        ],
        'Model Development': [
            '✓ Experiment tracking',
            '✓ Model version control',
            '✓ Hyperparameter tuning',
            '✓ Model evaluation metrics'
        ],
        'Deployment & Monitoring': [
            '✓ CI/CD pipeline for models',
            '✓ Model performance monitoring',
            '✓ Data drift detection',
            '✓ Automated retraining triggers'
        ],
        'Infrastructure': [
            '✓ Scalable serving infrastructure',
            '✓ Model registry',
            '✓ Monitoring dashboard',
            '✓ Alerting system'
        ]
    }
    
    print("\n=== MLOps Best Practices Checklist ===")
    for category, items in checklist.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")
    
    return checklist

def main():
    """Main MLOps demonstration"""
    print("=== MLOps Basics Demonstration ===\n")
    
    # Run MLOps pipeline
    model_v1, model_v2 = demonstrate_mlops_pipeline()
    
    # Show checklist
    checklist = create_mlops_checklist()
    
    print("\n=== MLOps Demo Complete ===")
    print("Key concepts demonstrated:")
    print("✓ Model training and versioning")
    print("✓ Performance tracking") 
    print("✓ Model artifact management")
    print("✓ Performance monitoring")
    print("✓ Model retraining")
    print("✓ Best practices checklist")
    
    return model_v1, model_v2, checklist

if __name__ == "__main__":
    model_v1, model_v2, checklist = main()