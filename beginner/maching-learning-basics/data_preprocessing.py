from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

# Create sample dataset with mixed data types
data = {
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 70000, 80000, 90000],
    'city': ['NYC', 'SF', 'NYC', 'Chicago', 'SF']
}

df = pd.DataFrame(data)

# Preprocessing
# Encode categorical variables
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

# Scale numerical features
scaler = StandardScaler()
df[['age_scaled', 'salary_scaled']] = scaler.fit_transform(df[['age', 'salary']])

print("Processed Data:")
print(df)