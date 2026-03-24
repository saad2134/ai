# Imports
import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB


# Importing the dataset
dataset = pd.read_csv('../datasets/Logistic_car_data.csv')
print("\n Dataset \n", dataset, "\n")

# input
x = dataset.iloc[:,[2,3]].values
print("\n Input Dataset \n", x, "\n")

#Target
y = dataset.iloc[:,4].values
print("\n Target Dataset \n", y, "\n")

#Plot of Age vs Purchased
x2 = dataset.iloc[:,[2]].values
plt.scatter(x2, y)
plt.xlabel("Age")
plt.ylabel("Purchased")
plt.title("Age vs Purchased")
plt.show()

#Plot of Salary vs Purchased
x3 = dataset.iloc[:,[3]].values
plt.scatter(x3, y)
plt.xlabel("Salary")
plt.ylabel("Purchased")
plt.title("Salary vs Purchased")
plt.show()

# Splitting the dataset into Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test) 
print("\n Predicted Values: \n", y_pred)


#Evaluation
acc = accuracy_score(y_test, y_pred)*100
print("\nAccuracy of Naive Bayes Classifier: ", acc)

#  Making Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix \n", cm, "\n")

# Visualizing Confusion Matrix
fig, ax = plt.subplots(figsize = (5,5))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1), ticklabels=('Predicted, Does not Buy Car', 'Predicted, Buys Car'))
ax.yaxis.set(ticks=(0,1), ticklabels=('Actual, Does not Buy Car', 'Actual, Buys Car'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.show()

#Visualizing the training set results
def visualize(x_set, y_set, mode): 
    X1, X2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max() + 1, step= 0.01),
                         nm.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 1].max() + 1, step= 0.01))
    plt.contourf(X1,X2, classifier.predict(
                     nm.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
                         cmap = ListedColormap(('white', 'black')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i,j in enumerate(nm.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                    color = ListedColormap(('purple','green'))(i), label = j)
    plt.title('Naive Bayes ('+mode+' Set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
visualize(x_train, y_train, "Training")

#Visualizing the test set results
visualize(x_test, y_test, "Test")

