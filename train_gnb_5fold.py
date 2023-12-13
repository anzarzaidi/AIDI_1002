# script to train VBL-VA001

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


def execute():
    X = pd.read_csv("data/existing/feature_VBL-VA001.csv", header=None)
    y = pd.read_csv("data/existing/label_VBL-VA001.csv", header=None)
    y = pd.Series.ravel(y)
    var_gnb = [10.0 ** i for i in np.arange(-1, -100, -1)]
    test_accuracy = np.empty(len(var_gnb))

    for i, k in enumerate(var_gnb):
        clf_gnb = GaussianNB(var_smoothing=k)
        scores = cross_val_score(clf_gnb, X, y, cv=5)
        print(scores)
        test_accuracy[i] = np.mean(scores)

    print(f"Max test acc: {np.max(test_accuracy)}")
    max_var_gnb = np.argmax(test_accuracy)
    print(f"Best var smoothing: {var_gnb[max_var_gnb]}")


execute()
