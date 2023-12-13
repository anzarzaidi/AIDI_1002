# script to train VBL-VA001

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def execute(plot):
    x = pd.read_csv('data/existing/feature_VBL-VA001.csv', header=None)
    y = pd.read_csv('data/existing/label_VBL-VA001.csv', header=None)
    y = pd.Series.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True)
    print("Shape of Train Data : {}".format(X_train.shape))
    print("Shape of Test Data : {}".format(X_test.shape))
    neighbors = np.arange(1, 100)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train, y_train)
        train_accuracy[i] = knn.score(X_train.values, y_train)
        test_accuracy[i] = knn.score(X_test.values, y_test)

    print(f"Max test acc: {np.max(test_accuracy)}")
    if plot:
       plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
       plt.plot(neighbors, train_accuracy, label='Training accuracy')
       plt.legend()
       plt.xlabel('Number of neighbors')
       plt.ylabel('Accuracy')
       plt.show()
    print(f"Optimal k: {np.argmax(test_accuracy)}")
    print(f"Max test accuracy: {max(test_accuracy)}")


execute(True)
