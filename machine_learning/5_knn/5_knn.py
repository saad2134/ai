# Statistical Implementation
import numpy as np
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]


training_data = [[1,2],[2,3],[3,4],[6,7],[7,8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = [8,9]
k = 4

prediction = knn_predict(training_data, training_labels, test_point, k)
print(prediction)
