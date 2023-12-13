# script to train VBL-VA001, 5-cv knn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


def execute():
    X = pd.read_csv('data/existing/feature_VBL-VA001.csv', header=None)
    y = pd.read_csv('data/existing/label_VBL-VA001.csv', header=None)
    y = pd.Series.ravel(y)
    neighbors = np.arange(1, 100)
    test_accuracy = np.empty(len(neighbors))
    for i, k in enumerate(neighbors):
        clf_knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf_knn, X.values, y, cv=5)
        test_accuracy[i] = np.mean(scores)
    print(f"Max test acc: {np.max(test_accuracy)}")
    print(f"Best neighbors: {np.argmax(test_accuracy) + 1}")


execute()
