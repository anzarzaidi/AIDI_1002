# script to train VBL-VA001

from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix




def execute(plot):
    x = pd.read_csv("data/existing/feature_VBL-VA001.csv", header=None)
    y = pd.read_csv("data/existing/label_VBL-VA001.csv", header=None)
    y = pd.Series.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )
    print('*************************************************************************')
    print("BernoulliNB Classifier")
    print("Shape of Train Data : {}".format(X_train.shape))
    print("Shape of Test Data : {}".format(X_test.shape))
    model = BernoulliNB()
    multiclass_classifier = OneVsRestClassifier(model)
    # Train the model
    multiclass_classifier.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = multiclass_classifier.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)
    print('Testing Set Evaluation F1-Score=>', f1_score(y_test, y_pred, average='macro'))
    print('*************************************************************************')


execute(True)