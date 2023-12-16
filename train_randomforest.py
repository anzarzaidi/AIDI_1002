# script to train VBL-VA001

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score



def execute():
    x = pd.read_csv("data/existing/feature_VBL-VA001.csv", header=None)
    y = pd.read_csv("data/existing/label_VBL-VA001.csv", header=None)
    y = pd.Series.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )
    print('*************************************************************************')
    print("RandomForestClassifier Classifier")
    print("Shape of Train Data : {}".format(X_train.shape))
    print("Shape of Test Data : {}".format(X_test.shape))
    rfc = RandomForestClassifier(criterion='entropy', random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(rfc.score(X_test, y_test)))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", conf_matrix)
    # Display classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)
    print('Testing Set Evaluation F1-Score=>', f1_score(y_test, y_pred, average='macro'))
    print('*************************************************************************')


execute()

