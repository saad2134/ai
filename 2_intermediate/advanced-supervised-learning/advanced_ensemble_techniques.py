import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def advanced_ensemble_comparison():
    """Compare advanced ensemble techniques"""
    # Create complex dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        flip_y=0.1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Advanced ensemble methods
    base_estimator = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    ensembles = {
        'Bagging (Decision Tree)': BaggingClassifier(
            estimator=base_estimator,
            n_estimators=50,
            random_state=42
        ),
        'AdaBoost (Decision Tree)': AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=50,
            random_state=42
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=100,
            random_state=42
        ),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42),
    }
    
    # Add Random Forest for comparison
    from sklearn.ensemble import RandomForestClassifier
    ensembles['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train and evaluate
    results = {}
    for name, ensemble in ensembles.items():
        print(f"Training {name}...")
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': ensemble,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    # Display results
    results_df = pd.DataFrame({
        'Ensemble Method': list(results.keys()),
        'Test Accuracy': [results[name]['accuracy'] for name in results.keys()],
        'CV Mean': [results[name]['cv_mean'] for name in results.keys()],
        'CV Std': [results[name]['cv_std'] for name in results.keys()]
    }).sort_values('Test Accuracy', ascending=False)
    
    print("\n=== Advanced Ensemble Methods Comparison ===")
    print(results_df.round(4))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(results_df))
    plt.bar(x_pos, results_df['Test Accuracy'], yerr=results_df['CV Std'],
            capsize=5, alpha=0.7, color='lightseagreen')
    plt.xticks(x_pos, results_df['Ensemble Method'], rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Advanced Ensemble Methods Performance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results, results_df

def bagging_vs_boosting_analysis():
    """Detailed analysis of Bagging vs Boosting"""
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
    
    # Different numbers of estimators
    n_estimators_range = [10, 25, 50, 100, 200]
    
    bagging_scores = []
    boosting_scores = []
    
    for n_est in n_estimators_range:
        # Bagging
        bagging = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_est,
            random_state=42
        )
        bagging.fit(X_train, y_train)
        bagging_scores.append(bagging.score(X_test, y_test))
        
        # Boosting (AdaBoost)
        boosting = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_est,
            random_state=42
        )
        boosting.fit(X_train, y_train)
        boosting_scores.append(boosting.score(X_test, y_test))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, bagging_scores, 'o-', label='Bagging', linewidth=2)
    plt.plot(n_estimators_range, boosting_scores, 'o-', label='Boosting (AdaBoost)', linewidth=2)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Test Accuracy')
    plt.title('Bagging vs Boosting: Effect of Ensemble Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Bagging vs Boosting Analysis:")
    print(f"Best Bagging score: {max(bagging_scores):.4f}")
    print(f"Best Boosting score: {max(boosting_scores):.4f}")
    
    return bagging_scores, boosting_scores

def ensemble_diversity_analysis():
    """Analyze diversity in ensemble predictions"""
    from scipy.stats import entropy
    from sklearn.metrics import pairwise_distances
    
    X, y = make_classification(n_samples=300, n_features=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create diverse base estimators
    estimators = [
        DecisionTreeClassifier(max_depth=5, random_state=42),
        DecisionTreeClassifier(max_depth=10, random_state=42),
        DecisionTreeClassifier(max_depth=None, random_state=42),
        SVC(probability=True, random_state=42),
        LogisticRegression(random_state=42, max_iter=1000)
    ]
    
    # Collect predictions from all estimators
    all_predictions = []
    for est in estimators:
        est.fit(X_train, y_train)
        pred_proba = est.predict_proba(X_test)[:, 1] if hasattr(est, 'predict_proba') else est.decision_function(X_test)
        all_predictions.append(pred_proba)
    
    all_predictions = np.array(all_predictions)
    
    # Calculate diversity measures
    # 1. Prediction correlation
    correlation_matrix = np.corrcoef(all_predictions)
    avg_correlation = (np.sum(np.abs(correlation_matrix)) - len(estimators)) / (len(estimators) * (len(estimators) - 1))
    
    # 2. Prediction disagreement
    binary_predictions = (all_predictions > 0.5).astype(int)
    disagreement = 1 - np.mean(binary_predictions == binary_predictions[0], axis=0).mean()
    
    print("=== Ensemble Diversity Analysis ===")
    print(f"Average correlation between predictors: {avg_correlation:.4f}")
    print(f"Average disagreement rate: {disagreement:.4f}")
    print(f"Number of diverse estimators: {len(estimators)}")
    
    # Plot correlation matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation between Ensemble Predictors')
    plt.xticks(range(len(estimators)), [f'Est{i+1}' for i in range(len(estimators))])
    plt.yticks(range(len(estimators)), [f'Est{i+1}' for i in range(len(estimators))])
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix, disagreement

if __name__ == "__main__":
    # 1. Compare advanced ensemble methods
    results, results_df = advanced_ensemble_comparison()
    
    # 2. Bagging vs Boosting analysis
    bagging_scores, boosting_scores = bagging_vs_boosting_analysis()
    
    # 3. Ensemble diversity analysis
    correlation_matrix, disagreement = ensemble_diversity_analysis()