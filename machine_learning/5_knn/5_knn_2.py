# sklearn Implementation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

f = "../datasets/Social_Network_Ads.csv"
dataset = pd.read_csv(f)

X = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, -1].values

print(len(X))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:,0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score, classification_report
print(y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print('--------------------')
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.figure(figsize=(6,5))
not_purchased_label = False
purchased_label = False
for i in range(len(X_test)):
    if y_pred[i] == 0:
        plt.scatter(X_test[i,1], X_test[i,2], color='red',
                    label='Not Purchased' if not not_purchased_label else "")
        not_purchased_label = True
    else:
        plt.scatter(X_test[i,1], X_test[i,2], color='green',
                    label='Purchased' if not purchased_label else "")
        purchased_label = True

        
plt.title("KNN Test Set Results")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
