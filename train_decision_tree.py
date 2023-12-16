# script to train VBL-VA001

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score,classification_report


def execute():
    x = pd.read_csv("data/existing/feature_VBL-VA001.csv", header=None)
    y = pd.read_csv("data/existing/label_VBL-VA001.csv", header=None)
    y = pd.Series.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )
    print('*************************************************************************')
    print("DecisionTree Classifier")
    print("Shape of Train Data : {}".format(X_train.shape))
    print("Shape of Test Data : {}".format(X_test.shape))
    dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt.fit(X_train, y_train)
    dt_pred_train = dt.predict(X_train)
    dt_pred_test = dt.predict(X_test)


    # Evaluate the model
    accuracy = accuracy_score(y_test, dt_pred_test)
    conf_matrix = confusion_matrix(y_test, dt_pred_test)
    class_report = classification_report(y_test, dt_pred_test)

    print("Accuracy:", accuracy)
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)
    print('Testing Set Evaluation F1-Score=>', f1_score(y_test, dt_pred_test, average='macro'))
    dt_pred_train = dt.predict(X_train)
    print('Training Set Evaluation F1-Score=>', f1_score(y_train, dt_pred_train, average='macro'))



execute()

