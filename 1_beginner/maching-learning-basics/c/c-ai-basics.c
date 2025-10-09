#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Simple Linear Regression in C
typedef struct {
    double slope;
    double intercept;
} LinearModel;

// Train linear regression model
LinearModel train_linear_regression(double x[], double y[], int n) {
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    
    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }
    
    LinearModel model;
    model.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    model.intercept = (sum_y - model.slope * sum_x) / n;
    
    return model;
}

// Predict using trained model
double predict(LinearModel model, double x) {
    return model.slope * x + model.intercept;
}

// Simple K-Nearest Neighbors in C
double euclidean_distance(double a[], double b[], int features) {
    double sum = 0;
    for (int i = 0; i < features; i++) {
        sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
}

int knn_predict(double train[][2], int labels[], int train_size, 
                double test[], int k, int features) {
    // Calculate distances to all training points
    double distances[train_size];
    for (int i = 0; i < train_size; i++) {
        distances[i] = euclidean_distance(train[i], test, features);
    }
    
    // Simple voting (for demo - in real implementation, sort and find k nearest)
    int count_0 = 0, count_1 = 0;
    for (int i = 0; i < k && i < train_size; i++) {
        if (labels[i] == 0) count_0++;
        else count_1++;
    }
    
    return (count_1 > count_0) ? 1 : 0;
}

// Simple Perceptron (Neural Network) in C
typedef struct {
    double weights[3]; // for 2 features + bias
    double learning_rate;
} Perceptron;

void initialize_perceptron(Perceptron *p) {
    srand(time(NULL));
    for (int i = 0; i < 3; i++) {
        p->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Random between -1 and 1
    }
    p->learning_rate = 0.1;
}

int perceptron_predict(Perceptron *p, double inputs[]) {
    double sum = p->weights[2]; // bias
    for (int i = 0; i < 2; i++) {
        sum += inputs[i] * p->weights[i];
    }
    return (sum > 0) ? 1 : 0;
}

void perceptron_train(Perceptron *p, double inputs[][2], int labels[], int epochs, int data_size) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < data_size; i++) {
            int prediction = perceptron_predict(p, inputs[i]);
            int error = labels[i] - prediction;
            
            // Update weights
            for (int j = 0; j < 2; j++) {
                p->weights[j] += p->learning_rate * error * inputs[i][j];
            }
            p->weights[2] += p->learning_rate * error; // bias update
        }
    }
}

int main() {
    printf("=== Basic AI in C ===\n");
    
    // Linear Regression Example
    printf("\n1. Linear Regression:\n");
    double x[] = {1, 2, 3, 4, 5};
    double y[] = {2, 4, 6, 8, 10};
    int n = 5;
    
    LinearModel model = train_linear_regression(x, y, n);
    printf("Model: y = %.2fx + %.2f\n", model.slope, model.intercept);
    printf("Prediction for x=6: %.2f\n", predict(model, 6));
    
    // KNN Example
    printf("\n2. K-Nearest Neighbors:\n");
    double train_data[][2] = {{1, 2}, {2, 3}, {3, 1}, {4, 2}};
    int train_labels[] = {0, 0, 1, 1};
    double test_point[] = {2.5, 2};
    
    int prediction = knn_predict(train_data, train_labels, 4, test_point, 3, 2);
    printf("KNN Prediction: %d\n", prediction);
    
    // Perceptron Example
    printf("\n3. Perceptron (Neural Network):\n");
    Perceptron p;
    initialize_perceptron(&p);
    
    double perceptron_data[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int and_labels[] = {0, 0, 0, 1}; // AND gate
    
    perceptron_train(&p, perceptron_data, and_labels, 10, 4);
    
    printf("AND Gate Predictions:\n");
    for (int i = 0; i < 4; i++) {
        int pred = perceptron_predict(&p, perceptron_data[i]);
        printf("  [%.0f, %.0f] -> %d (expected %d)\n", 
               perceptron_data[i][0], perceptron_data[i][1], pred, and_labels[i]);
    }
    
    return 0;
}