import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def create_complex_dataset():
    """Create a complex classification dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        flip_y=0.05,
        random_state=42
    )
    
    return X, y

def compare_ensemble_methods():
    """Compare different ensemble methods"""
    X, y = create_complex_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Define ensemble models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42),
    }
    
    # Individual classifiers for voting
    lr = LogisticRegression(random_state=42, max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(probability=True, random_state=42)
    
    # Voting classifier
    voting_clf = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('svm', svm)],
        voting='soft'
    )
    models['Voting Classifier'] = voting_clf
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
    
    # Display results
    print("\n=== Ensemble Methods Comparison ===")
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test Accuracy': [results[name]['accuracy'] for name in results.keys()],
        'CV Mean': [results[name]['cv_mean'] for name in results.keys()],
        'CV Std': [results[name]['cv_std'] for name in results.keys()]
    }).sort_values('Test Accuracy', ascending=False)
    
    print(results_df.round(4))
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(results_df))
    plt.bar(x_pos, results_df['Test Accuracy'], yerr=results_df['CV Std'], 
            capsize=5, alpha=0.7, color='skyblue')
    plt.xticks(x_pos, results_df['Model'], rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Ensemble Methods Performance Comparison')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results, results_df

def random_forest_feature_importance():
    """Analyze feature importance in Random Forest"""
    X, y = create_complex_dataset()
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest - Feature Importance")
    plt.bar(range(X.shape[1]), importance[indices])
    plt.xticks(range(X.shape[1]), indices)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
    
    print("Top 5 most important features:")
    for i in range(5):
        print(f"Feature {indices[i]}: {importance[indices[i]]:.4f}")
    
    return importance, indices

def gradient_boosting_analysis():
    """Analyze Gradient Boosting behavior"""
    X, y = create_complex_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train GB with different numbers of estimators
    n_estimators_range = [10, 50, 100, 200, 500]
    train_scores = []
    test_scores = []
    
    for n_est in n_estimators_range:
        gb = GradientBoostingClassifier(
            n_estimators=n_est,
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(X_train, y_train)
        
        train_score = gb.score(X_train, y_train)
        test_score = gb.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_scores, 'o-', label='Training Score')
    plt.plot(n_estimators_range, test_scores, 'o-', label='Test Score')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Gradient Boosting: Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return train_scores, test_scores

def xgboost_lightgbm_comparison():
    """Detailed comparison between XGBoost and LightGBM"""
    X, y = create_complex_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Hyperparameter tuning for XGBoost
    xgb_params = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    
    # Hyperparameter tuning for LightGBM
    lgb_params = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'num_leaves': [31, 63]
    }
    
    lgb = LGBMClassifier(random_state=42)
    lgb_grid = GridSearchCV(lgb, lgb_params, cv=3, scoring='accuracy', n_jobs=-1)
    lgb_grid.fit(X_train, y_train)
    
    print("=== XGBoost vs LightGBM ===")
    print(f"XGBoost Best Params: {xgb_grid.best_params_}")
    print(f"XGBoost Best CV Score: {xgb_grid.best_score_:.4f}")
    print(f"XGBoost Test Score: {xgb_grid.best_estimator_.score(X_test, y_test):.4f}")
    
    print(f"\nLightGBM Best Params: {lgb_grid.best_params_}")
    print(f"LightGBM Best CV Score: {lgb_grid.best_score_:.4f}")
    print(f"LightGBM Test Score: {lgb_grid.best_estimator_.score(X_test, y_test):.4f}")
    
    # Compare training time
    import time
    
    # XGBoost training time
    start_time = time.time()
    xgb_final = XGBClassifier(**xgb_grid.best_params_, random_state=42)
    xgb_final.fit(X_train, y_train)
    xgb_time = time.time() - start_time
    
    # LightGBM training time
    start_time = time.time()
    lgb_final = LGBMClassifier(**lgb_grid.best_params_, random_state=42)
    lgb_final.fit(X_train, y_train)
    lgb_time = time.time() - start_time
    
    print(f"\nTraining Time Comparison:")
    print(f"XGBoost: {xgb_time:.4f} seconds")
    print(f"LightGBM: {lgb_time:.4f} seconds")
    print(f"Speedup: {xgb_time/lgb_time:.2f}x")
    
    return xgb_grid.best_estimator_, lgb_grid.best_estimator_

def ensemble_stacking():
    """Implement stacking ensemble"""
    from sklearn.ensemble import StackingClassifier
    
    X, y = create_complex_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define base estimators
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('svm', SVC(probability=True, random_state=42))
    ]
    
    # Define stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    
    # Train and evaluate
    stacking_clf.fit(X_train, y_train)
    stacking_score = stacking_clf.score(X_test, y_test)
    
    # Compare with individual models
    individual_scores = {}
    for name, estimator in base_estimators:
        estimator.fit(X_train, y_train)
        individual_scores[name] = estimator.score(X_test, y_test)
    
    print("\n=== Stacking Ensemble Results ===")
    for name, score in individual_scores.items():
        print(f"{name}: {score:.4f}")
    print(f"Stacking Classifier: {stacking_score:.4f}")
    
    # Plot comparison
    models = list(individual_scores.keys()) + ['Stacking']
    scores = list(individual_scores.values()) + [stacking_score]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.ylabel('Accuracy')
    plt.title('Stacking Ensemble vs Individual Models')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return stacking_clf, individual_scores

# Run all analyses
if __name__ == "__main__":
    # 1. Compare ensemble methods
    results, results_df = compare_ensemble_methods()
    
    # 2. Feature importance analysis
    importance, indices = random_forest_feature_importance()
    
    # 3. Gradient boosting analysis
    train_scores, test_scores = gradient_boosting_analysis()
    
    # 4. XGBoost vs LightGBM
    best_xgb, best_lgb = xgboost_lightgbm_comparison()
    
    # 5. Stacking ensemble
    stacking_model, individual_scores = ensemble_stacking()