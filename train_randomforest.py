# script to train VBL-VA001

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def execute():
    x = pd.read_csv("data/existing/feature_VBL-VA001.csv", header=None)
    y = pd.read_csv("data/existing/label_VBL-VA001.csv", header=None)
    y = pd.Series.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )
    print("Shape of Train Data : {}".format(X_train.shape))
    print("Shape of Test Data : {}".format(X_test.shape))
    var_gnb = [10.0 ** i for i in np.arange(-1, -100, -1)]
    train_accuracy = np.empty(len(var_gnb))
    test_accuracy = np.empty(len(var_gnb))

    for i, k in enumerate(var_gnb):
        rfc = RandomForestClassifier(criterion='entropy', random_state=42)
        gnb = rfc.fit(X_train, y_train)
        train_accuracy[i] = gnb.score(X_train, y_train)
        test_accuracy[i] = gnb.score(X_test, y_test)

    print(f"Max test acc: {np.max(test_accuracy)}")
    print(f"Optimal var_gnb: {np.argmax(test_accuracy)}")
    print(f"Max test accuracy: {max(test_accuracy)}")


execute()

