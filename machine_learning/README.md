# Machine Learning

A collection of machine learning algorithms and exercises covering fundamental ML concepts.

## Directory Structure

```
machine_learning/
├── datasets/                    # Shared CSV data files
├── 2_numpy_pandas_scipy/       # NumPy, Pandas, SciPy exercises
├── 3_linear_regression/       # Linear regression implementations
├── 3_logistic_regression/      # Logistic regression
├── 4_decision_trees/          # Decision tree classifiers
├── 5_knn/                     # K-Nearest Neighbors
├── 6_naives_bayes/            # Naive Bayes classifier
└── 7_support_vector_machine/  # SVM classifier
```

## Datasets

| File | Description |
|------|-------------|
| `diabetes.csv` | Pima Indians Diabetes Dataset |
| `Logistic_car_data.csv` | Car purchase data |
| `Logistic_Iris.csv` | Iris flower dataset |
| `Social_Network_Ads.csv` | Social network ad data |

## Programs

### 2_numpy_pandas_scipy
- `2_numpy_pandas_scipy.py` - Data manipulation with NumPy, Pandas, SciPy
- `2_numpy_pandas_scipy_output.txt` - Sample output

### 3_linear_regression
- `3_linear_regression.py` - Linear regression implementation
- `3_linear_regression_2.py` - Linear regression (alternative)
- `3_linear_regression_output.txt` - Sample output
- `3_linear_regression_2_output.txt` - Sample output
- `3a_2.txt`, `3b.txt` - Notes
- `3a_1_Figure_1.png` - Visualization

### 3_logistic_regression
- `3_logistic_regression.py` - Logistic regression for classification
- `3_logistic_regression_output.txt` - Sample output

### 4_decision_trees
- `4_decision_trees.py` - Decision tree classifier
- `4_decision_trees_2.py` - Decision trees with diabetes dataset
- `4_decision_trees_b.py` - Decision trees (alternative)
- `4_decision_trees_output.txt` - Sample output
- `4_decision_trees_2_output.txt` - Sample output
- `4_decision_trees_b_output.txt` - Sample output
- `4_2_Figure_1.png` - Visualization

### 5_knn
- `5_knn.py` - K-Nearest Neighbors algorithm
- `5_knn_2.py` - KNN with social network ads data
- `5_knn_output.txt` - Sample output
- `5_knn_2_output.txt` - Sample output

### 6_naives_bayes
- `6_naives_bayes.py` - Naive Bayes classifier
- `6_naives_bayes_output.txt` - Sample output

### 7_support_vector_machine
- `7_support_vector_machine.py` - SVM classifier
- `7_support_vector_machine_output.txt` - Sample output

## Running the Scripts

```bash
cd <program_folder>
python <script_name>.py
```

Scripts read data from `../datasets/` and output to console.
