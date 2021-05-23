#created and edited by Samuel Phillips

#imports for data, classes and more
from sklearn.datasets import load_iris
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#-- a4p1 starts here --
#iris data is loaded
iris = load_iris()
iData = iris.data

#data is collected from the iris dataset
X = iris.data[50:]
y = iris.target[50:]

#leave one out is used to split the data from above
ol = LeaveOneOut()
num_cases = ol.get_n_splits(X)
ttsi = ol.split(X)

#error rates list is created to hold all the averages of the error rates
error_rates = []

#train test cycle that cycles through 100 times with the MLP classifier
train_test_cycle = 1
for train_index, test_index in ttsi:
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    clf = MLPClassifier(random_state=1, hidden_layer_sizes=50, max_iter=1000).fit(X_train, y_train)
    p1 = clf.predict(X_test)
    
    vals = []
    
    for z in range(0, len(p1)):
        if p1[z] == y_test[z]:
            vals.append(0)

        elif p1[z] != y_test[z]:
            vals.append(1)
    
    error_rates.append(np.mean(vals))
    
    train_test_cycle += 1
    
#overall ANN average error rates is calculated and printed
print('ANN Average Error Rate: ' + str(np.average(error_rates)))
#-- a4p1 ends here --

#-- a4p2 starts here --
#iris data is loaded
iris = load_iris()
iData = iris.data

#data is collected from the iris dataset
X = iris.data[50:]
y = iris.target[50:]

#leave one out is used to split the data from above
ol = LeaveOneOut()
num_cases = ol.get_n_splits(X)
ttsi = ol.split(X)

#empty list to hold error values are created
e1 = []
e2 = []

#train test cycle that cycles through 100 times with the decision tree classifier and
#the kneighbors classifier
train_test_cycle = 1
for train_index, test_index in ttsi:
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    p1 = clf.predict(X_test)

    v1 = []
    
    for z in range(0, len(p1)):
        if p1[z] == y_test[z]:
            v1.append(0)

        elif p1[z] != y_test[z]:
            v1.append(1)
    
    e1.append(np.mean(v1))
    
    clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    p2 = clf.predict(X_test)

    v2 = []
    
    for z in range(0, len(p1)):
        if p2[z] == y_test[z]:
            v2.append(0)

        elif p2[z] != y_test[z]:
            v2.append(1)
    
    e2.append(np.mean(v2))
    
    train_test_cycle += 1
    

#overall Decision Tree Average average error rates is calculated and printed
print('Decision Tree Average Error Rate: ' + str(np.average(e1)))

#overall KNN average error rates is calculated and printed
print('KNN Average Error Rate: ' + str(np.average(e2)))
#-- a4p2 ends here --