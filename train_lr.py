import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def execute():
    global confusion_matrix
    x = pd.read_csv("data/existing/feature_VBL-VA001.csv", header=None)
    y = pd.read_csv("data/existing/label_VBL-VA001.csv", header=None)
    y = pd.Series.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    logreg = LogisticRegression()
    logreg.fit(X_scaled, y_train)
    y_pred = logreg.predict(X_test)
    print('*************************************************************************')
    print("LogisticRegression Classifier")
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", conf_matrix)
    # Display classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)
    print('Testing Set Evaluation F1-Score=>', f1_score(y_test, y_pred, average='macro'))
    print('*************************************************************************')

    print('*************************************************************************')


execute()