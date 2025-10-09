from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load multi-class dataset
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - Digit Classification')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Detailed analysis
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Calculate per-class metrics
class_report = classification_report(y_test, y_pred, output_dict=True)
class_metrics = pd.DataFrame(class_report).transpose()
print("\n=== Per-Class Metrics ===")
print(class_metrics.round(4))

# Find most confused pairs
misclassified = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i, j] > 0:
            misclassified.append((i, j, cm[i, j]))

misclassified.sort(key=lambda x: x[2], reverse=True)
print("\n=== Most Common Misclassifications ===")
for actual, predicted, count in misclassified[:5]:
    print(f"Actual: {actual} → Predicted: {predicted} (Count: {count})")