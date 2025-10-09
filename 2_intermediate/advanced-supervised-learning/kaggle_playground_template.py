import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

class KagglePlaygroundSolver:
    """Template for Kaggle playground competitions"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data (template method)"""
        # This is a template - replace with actual data loading
        from sklearn.datasets import make_classification
        
        # Create sample competition data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=1,
            flip_y=0.05,
            random_state=42
        )
        
        # Simulate missing values (common in Kaggle)
        mask = np.random.random(X.shape) < 0.05
        X[mask] = np.nan
        
        # Convert to DataFrame to simulate real dataset
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        return X_df, y
    
    def preprocess_data(self, X, y):
        """Comprehensive data preprocessing"""
        # Handle missing values
        X_processed = X.copy()
        
        # Fill numerical missing values with median
        for col in X_processed.select_dtypes(include=[np.number]).columns:
            X_processed[col].fillna(X_processed[col].median(), inplace=True)
        
        # Feature engineering
        X_processed = self.create_features(X_processed)
        
        return X_processed, y
    
    def create_features(self, X):
        """Create new features (feature engineering)"""
        X_eng = X.copy()
        
        # Example feature engineering techniques
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        # Statistical features
        X_eng['mean_features'] = X[numerical_cols].mean(axis=1)
        X_eng['std_features'] = X[numerical_cols].std(axis=1)
        X_eng['max_features'] = X[numerical_cols].max(axis=1)
        X_eng['min_features'] = X[numerical_cols].min(axis=1)
        
        # Interaction features
        if len(numerical_cols) >= 2:
            X_eng['feature_interaction_1'] = X[numerical_cols[0]] * X[numerical_cols[1]]
        
        # Polynomial features (simple version)
        for col in numerical_cols[:3]:  # Only first 3 columns to avoid explosion
            X_eng[f'{col}_squared'] = X[col] ** 2
        
        return X_eng
    
    def initialize_models(self):
        """Initialize multiple models for comparison"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=200, random_state=42),
        }
    
    def tune_hyperparameters(self, X_train, y_train):
        """Tune hyperparameters for the best model"""
        print("Tuning hyperparameters for XGBoost...")
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0]
        }
        
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """Evaluate all models"""
        self.results = {}
        
        for name, model in self.models.items():
            print(f"Training and evaluating {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        return self.results
    
    def display_results(self):
        """Display comparison results"""
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test Accuracy': [self.results[name]['accuracy'] for name in self.results.keys()],
            'CV Mean': [self.results[name]['cv_mean'] for name in self.results.keys()],
            'CV Std': [self.results[name]['cv_std'] for name in self.results.keys()]
        }).sort_values('Test Accuracy', ascending=False)
        
        print("\n=== Model Comparison Results ===")
        print(results_df.round(4))
        
        # Best model
        best_model_name = results_df.iloc[0]['Model']
        best_accuracy = results_df.iloc[0]['Test Accuracy']
        print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}")
        
        return results_df
    
    def create_submission_file(self, model, X_test, test_ids=None, filename='submission.csv'):
        """Create Kaggle submission file"""
        predictions = model.predict(X_test)
        
        if test_ids is None:
            test_ids = range(len(predictions))
        
        submission_df = pd.DataFrame({
            'Id': test_ids,
            'Prediction': predictions
        })
        
        submission_df.to_csv(filename, index=False)
        print(f"Submission file saved as {filename}")
        
        return submission_df
    
    def run_competition_pipeline(self):
        """Complete competition pipeline"""
        print("Starting Kaggle Playground Competition Pipeline...")
        
        # 1. Load and preprocess data
        print("Step 1: Loading and preprocessing data...")
        X, y = self.load_and_preprocess_data()
        X_processed, y = self.preprocess_data(X, y)
        
        # 2. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=42
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # 3. Initialize models
        print("Step 2: Initializing models...")
        self.initialize_models()
        
        # 4. Evaluate models
        print("Step 3: Training and evaluating models...")
        results = self.evaluate_models(X_train, X_test, y_train, y_test)
        
        # 5. Display results
        print("Step 4: Analyzing results...")
        results_df = self.display_results()
        
        # 6. Hyperparameter tuning for best model
        print("Step 5: Hyperparameter tuning...")
        best_model_name = results_df.iloc[0]['Model']
        if best_model_name == 'XGBoost':
            tuned_model = self.tune_hyperparameters(X_train, y_train)
            self.models['XGBoost_Tuned'] = tuned_model
            
            # Evaluate tuned model
            tuned_model.fit(X_train, y_train)
            tuned_accuracy = tuned_model.score(X_test, y_test)
            print(f"Tuned XGBoost accuracy: {tuned_accuracy:.4f}")
        
        # 7. Create submission
        print("Step 6: Creating submission file...")
        best_model = self.results[best_model_name]['model']
        submission = self.create_submission_file(best_model, X_test)
        
        print("\n=== Pipeline Complete ===")
        return results_df, submission

# Example usage for specific competition types
class TitanicSolver(KagglePlaygroundSolver):
    """Specialized solver for Titanic-like competitions"""
    
    def load_and_preprocess_data(self):
        # Titanic-specific data loading and preprocessing
        # This is a simplified version
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=1000,
            n_features=8,  # Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, etc.
            n_informative=6,
            random_state=42
        )
        
        feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Simulate Titanic-like data characteristics
        X_df['Age'] = np.abs(X_df['Age'] * 20 + 20)  # Age between 0-80
        X_df['Fare'] = np.abs(X_df['Fare'] * 50 + 10)  # Positive fare values
        
        return X_df, y
    
    def create_features(self, X):
        """Titanic-specific feature engineering"""
        X_eng = X.copy()
        
        # Family size
        X_eng['FamilySize'] = X_eng['SibSp'] + X_eng['Parch'] + 1
        
        # Is alone
        X_eng['IsAlone'] = (X_eng['FamilySize'] == 1).astype(int)
        
        # Age groups
        X_eng['AgeGroup'] = pd.cut(X_eng['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[1, 2, 3, 4, 5])
        
        # Fare per person
        X_eng['FarePerPerson'] = X_eng['Fare'] / X_eng['FamilySize']
        
        # Title extraction (simplified)
        X_eng['HasTitle'] = (X_eng['Sex'] > 0).astype(int)  # Simplified version
        
        return X_eng

# Run the competition pipeline
if __name__ == "__main__":
    # General competition solver
    print("=== General Kaggle Playground Solver ===")
    solver = KagglePlaygroundSolver()
    results_df, submission = solver.run_competition_pipeline()
    
    # Titanic-specific solver
    print("\n" + "="*50)
    print("=== Titanic Competition Solver ===")
    titanic_solver = TitanicSolver()
    titanic_results, titanic_submission = titanic_solver.run_competition_pipeline()