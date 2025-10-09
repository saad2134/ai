import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    """End-to-end ML pipeline from raw data to trained model"""
    
    def __init__(self):
        self.pipeline = None
        self.preprocessor = None
        self.model = None
        self.feature_names = None
    
    def generate_sample_data(self, n_samples=1000):
        """Generate sample raw data with realistic issues"""
        np.random.seed(42)
        
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(50000, 20000, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'employment_status': np.random.choice(['Employed', 'Unemployed', 'Self-Employed', 'Student'], n_samples),
            'loan_amount': np.random.exponential(10000, n_samples),
            'loan_duration': np.random.randint(12, 60, n_samples),
            'default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Target
        }
        
        df = pd.DataFrame(data)
        
        # Introduce realistic data issues
        df = self._introduce_data_issues(df)
        
        return df
    
    def _introduce_data_issues(self, df):
        """Introduce common data issues found in real datasets"""
        # Add missing values
        missing_indices = np.random.choice(len(df), 100, replace=False)
        df.loc[missing_indices[:25], 'income'] = np.nan
        df.loc[missing_indices[25:50], 'credit_score'] = np.nan
        df.loc[missing_indices[50:75], 'education'] = np.nan
        
        # Add outliers
        outlier_indices = np.random.choice(len(df), 20, replace=False)
        df.loc[outlier_indices[:10], 'income'] *= 10  # Extreme high income
        df.loc[outlier_indices[10:], 'age'] = 150     # Impossible age
        
        # Add inconsistent formatting
        df.loc[df.sample(50).index, 'employment_status'] = df.loc[df.sample(50).index, 'employment_status'].str.upper()
        
        return df
    
    def data_validation(self, df):
        """Validate data quality and identify issues"""
        print("=== DATA VALIDATION ===")
        
        validation_report = {}
        
        # Basic info
        validation_report['shape'] = df.shape
        validation_report['dtypes'] = df.dtypes.to_dict()
        
        # Missing values
        missing_data = df.isnull().sum()
        validation_report['missing_values'] = missing_data[missing_data > 0].to_dict()
        
        # Data quality checks
        validation_report['issues'] = []
        
        # Check for outliers in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in ['age', 'income', 'credit_score']:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
                if len(outliers) > 0:
                    validation_report['issues'].append(f"Outliers in {col}: {len(outliers)}")
        
        # Check for impossible values
        if (df['age'] > 120).any():
            validation_report['issues'].append("Impossible age values (>120)")
        if (df['credit_score'] < 300).any() or (df['credit_score'] > 850).any():
            validation_report['issues'].append("Credit score outside valid range")
        
        # Print validation report
        for key, value in validation_report.items():
            print(f"{key}: {value}")
        
        return validation_report
    
    def build_preprocessor(self, df):
        """Build data preprocessing pipeline"""
        print("\n=== BUILDING PREPROCESSOR ===")
        
        # Separate features and target
        X = df.drop(['customer_id', 'default'], axis=1)
        y = df['default']
        self.feature_names = X.columns.tolist()
        
        # Define numerical and categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical features: {numerical_cols}")
        print(f"Categorical features: {categorical_cols}")
        
        # Preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', LabelEncoder())  # Simplified for demo
        ])
        
        # Combine preprocessors
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        return self.preprocessor
    
    def build_models(self):
        """Define multiple ML models to compare"""
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(random_state=42)
        }
        return models
    
    def create_pipeline(self, model):
        """Create complete ML pipeline"""
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', model)
        ])
        return pipeline
    
    def train_and_evaluate(self, df):
        """Train and evaluate multiple models"""
        print("\n=== MODEL TRAINING & EVALUATION ===")
        
        # Prepare data
        X = df.drop(['customer_id', 'default'], axis=1)
        y = df['default']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Build preprocessor
        self.build_preprocessor(df)
        
        # Train and evaluate models
        models = self.build_models()
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline
            pipeline = self.create_pipeline(model)
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'pipeline': pipeline,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Detailed report for the best model
            if name == 'Random Forest':  # Typically performs well
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
        
        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.pipeline = results[best_model_name]['pipeline']
        self.model = models[best_model_name]
        
        print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
        
        return results
    
    def save_pipeline(self, filepath='ml_pipeline.joblib'):
        """Save the trained pipeline"""
        if self.pipeline is not None:
            joblib.dump(self.pipeline, filepath)
            print(f"\nPipeline saved to {filepath}")
        else:
            print("No pipeline to save. Train a model first.")
    
    def load_pipeline(self, filepath='ml_pipeline.joblib'):
        """Load a saved pipeline"""
        try:
            self.pipeline = joblib.load(filepath)
            print(f"Pipeline loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"File {filepath} not found.")
            return False
    
    def predict_new_data(self, new_data):
        """Make predictions on new data"""
        if self.pipeline is not None:
            predictions = self.pipeline.predict(new_data)
            probabilities = self.pipeline.predict_proba(new_data)
            
            results = pd.DataFrame({
                'prediction': predictions,
                'probability_class_0': probabilities[:, 0],
                'probability_class_1': probabilities[:, 1]
            })
            
            return results
        else:
            print("No trained pipeline available.")
            return None
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline from raw data to trained model"""
        print("=== COMPLETE ML PIPELINE ===")
        
        # Step 1: Generate/Load raw data
        print("\n1. Loading raw data...")
        raw_data = self.generate_sample_data(1000)
        print(f"Raw data shape: {raw_data.shape}")
        
        # Step 2: Data validation
        print("\n2. Validating data quality...")
        validation_report = self.data_validation(raw_data)
        
        # Step 3: Train and evaluate models
        print("\n3. Training models...")
        results = self.train_and_evaluate(raw_data)
        
        # Step 4: Save pipeline
        print("\n4. Saving pipeline...")
        self.save_pipeline()
        
        # Step 5: Demonstrate prediction on new data
        print("\n5. Testing pipeline on new data...")
        new_samples = self.generate_sample_data(5).drop(['customer_id', 'default'], axis=1)
        predictions = self.predict_new_data(new_samples)
        
        if predictions is not None:
            print("Predictions on new data:")
            print(predictions)
        
        return raw_data, results

def main():
    """Main function to demonstrate the complete ML pipeline"""
    pipeline = MLPipeline()
    raw_data, results = pipeline.run_complete_pipeline()
    
    print("\n=== PIPELINE EXECUTION COMPLETE ===")
    print("✓ Data validation completed")
    print("✓ Models trained and evaluated") 
    print("✓ Best model selected")
    print("✓ Pipeline saved")
    print("✓ Predictions demonstrated")
    
    return pipeline, raw_data, results

if __name__ == "__main__":
    pipeline, raw_data, results = main()