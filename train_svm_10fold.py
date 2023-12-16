# Cross validation 5 folds SVM evaluation
# Compare this snippet from train_svm.py:

from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score


def execute():
    X = pd.read_csv("data/existing/feature_VBL-VA001.csv", header=None)
    y = pd.read_csv("data/existing/label_VBL-VA001.csv", header=None)
    y = pd.Series.ravel(y)
    c_svm = np.arange(1, 100)
    test_accuracy = np.empty(len(c_svm))
    for i, k in enumerate(c_svm):
        clf_svm = SVC(C=k)
        scores = cross_val_score(clf_svm, X, y, cv=10)
        #print(scores)
        test_accuracy[i] = np.mean(scores)

    print('*************************************************************************')
    print("SVM 10 fold Classifier")
    print(f"Max test acc: {np.max(test_accuracy)}")
    print(f"Best C: {np.argmax(test_accuracy) + 1}")
    print('*************************************************************************')


execute()

