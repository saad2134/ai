import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Comprehensive feature engineering pipeline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
    
    def load_sample_data(self):
        """Create sample dataset for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(50000, 20000, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'city': np.random.choice(['New York', 'London', 'Tokyo', 'Paris'], n_samples),
            'purchase_amount': np.random.exponential(100, n_samples),
            'customer_since': np.random.randint(0, 365*5, n_samples),  # days
            'target': np.random.randint(0, 2, n_samples)
        }
        
        # Add some missing values
        missing_indices = np.random.choice(n_samples, 50, replace=False)
        data['income'][missing_indices[:25]] = np.nan
        data['age'][missing_indices[25:]] = np.nan
        
        return pd.DataFrame(data)
    
    def handle_missing_data(self, df):
        """Handle missing values using multiple strategies"""
        print("Handling missing data...")
        
        # Numerical columns - impute with median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                print(f"  Imputed {col} with median: {df[col].median():.2f}")
        
        # Categorical columns - impute with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"  Imputed {col} with mode: {mode_val}")
        
        return df
    
    def create_numerical_features(self, df):
        """Create new numerical features"""
        print("Creating numerical features...")
        
        # Binning continuous variables
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                                labels=['Young', 'Adult', 'Middle', 'Senior'])
        
        # Mathematical transformations
        df['log_income'] = np.log1p(df['income'])
        df['income_squared'] = df['income'] ** 2
        df['income_sqrt'] = np.sqrt(df['income'])
        
        # Ratio features
        df['purchase_per_day'] = df['purchase_amount'] / (df['customer_since'] + 1)
        
        # Statistical features
        df['income_zscore'] = (df['income'] - df['income'].mean()) / df['income'].std()
        
        return df
    
    def create_categorical_features(self, df):
        """Create categorical features and encodings"""
        print("Creating categorical features...")
        
        # Frequency encoding
        df['city_freq'] = df['city'].map(df['city'].value_counts())
        
        # Target encoding (mean encoding)
        city_target_mean = df.groupby('city')['target'].mean()
        df['city_target_encoded'] = df['city'].map(city_target_mean)
        
        # One-hot encoding for ML models
        df = pd.get_dummies(df, columns=['education', 'age_group'], prefix=['edu', 'age'])
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        print("Creating interaction features...")
        
        # Simple interactions
        df['age_income_interaction'] = df['age'] * df['income']
        df['income_purchase_ratio'] = df['income'] / (df['purchase_amount'] + 1)
        
        # Polynomial features for numerical columns
        numerical_cols = ['age', 'income', 'purchase_amount']
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(df[numerical_cols])
        poly_df = pd.DataFrame(poly_features, 
                              columns=poly.get_feature_names_out(numerical_cols))
        
        # Remove duplicate columns and join
        df = pd.concat([df, poly_df], axis=1)
        
        return df
    
    def create_time_based_features(self, df):
        """Create time-based features"""
        print("Creating time-based features...")
        
        # Convert days to years
        df['customer_years'] = df['customer_since'] / 365
        
        # Time-based categories
        df['customer_segment'] = pd.cut(df['customer_years'], 
                                       bins=[0, 1, 3, 5, 10],
                                       labels=['New', 'Regular', 'Loyal', 'VIP'])
        
        # Seasonal features (if we had dates)
        # df['month'] = pd.to_datetime(df['date']).dt.month
        # df['is_weekend'] = pd.to_datetime(df['date']).dt.dayofweek >= 5
        
        return df
    
    def select_features(self, X, y, k=20):
        """Select top k features using statistical tests"""
        print(f"Selecting top {k} features...")
        
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        selected_features = X.columns[selected_mask]
        
        print(f"Selected {len(selected_features)} features:")
        for feature in selected_features:
            print(f"  - {feature}")
        
        return X_selected, selected_features
    
    def evaluate_feature_importance(self, X, y):
        """Evaluate feature importance using Random Forest"""
        print("Evaluating feature importance...")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 most important features:")
        print(importance_df.head(10))
        
        return importance_df
    
    def run_pipeline(self, df):
        """Run complete feature engineering pipeline"""
        print("Starting feature engineering pipeline...")
        print(f"Original shape: {df.shape}")
        
        # Handle missing data
        df_clean = self.handle_missing_data(df.copy())
        
        # Create various feature types
        df_features = self.create_numerical_features(df_clean)
        df_features = self.create_categorical_features(df_features)
        df_features = self.create_interaction_features(df_features)
        df_features = self.create_time_based_features(df_features)
        
        print(f"Final shape: {df_features.shape}")
        print(f"Number of features created: {df_features.shape[1] - df.shape[1]}")
        
        return df_features

def main():
    """Main function to demonstrate feature engineering"""
    print("=== Feature Engineering Demonstration ===\n")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Load sample data
    df = engineer.load_sample_data()
    print("Original data:")
    print(df.head())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Run feature engineering pipeline
    df_engineered = engineer.run_pipeline(df)
    
    # Prepare data for modeling
    X = df_engineered.drop('target', axis=1)
    y = df_engineered['target']
    
    # Convert categorical columns to numeric
    X = pd.get_dummies(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Evaluate feature importance
    importance_df = engineer.evaluate_feature_importance(X_train, y_train)
    
    # Feature selection
    X_selected, selected_features = engineer.select_features(X_train, y_train, k=15)
    
    # Compare model performance
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_full.fit(X_train, y_train)
    y_pred_full = rf_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)
    
    rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
    X_test_selected = engineer.feature_selector.transform(X_test)
    rf_selected.fit(X_selected, y_train)
    y_pred_selected = rf_selected.predict(X_test_selected)
    acc_selected = accuracy_score(y_test, y_pred_selected)
    
    print(f"\nModel Performance Comparison:")
    print(f"Full features ({X_train.shape[1]}): {acc_full:.4f}")
    print(f"Selected features ({X_selected.shape[1]}): {acc_selected:.4f}")
    
    return engineer, df_engineered, importance_df

if __name__ == "__main__":
    engineer, df_engineered, importance_df = main()