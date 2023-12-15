import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


def execute():
    rfc = RandomForestClassifier()
    parameters = {
        'n_estimators': [5, 50, 250],
        'max_depth': [2, 4, 8, 16, 32, None]
    }
    print('*************************************************************************')
    print("RandomForest Classifier")
    x = pd.read_csv("data/existing/feature_VBL-VA001.csv", header=None)
    y = pd.read_csv("data/existing/label_VBL-VA001.csv", header=None)
    y = pd.Series.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )
    print("Shape of Train Data : {}".format(X_train.shape))
    print("Shape of Test Data : {}".format(X_test.shape))
    cv = GridSearchCV(rfc, parameters, cv=5)
    cv.fit(X_train, y_train)
    print_results(cv)
    rfc_predict = cv.predict(X_test)
    print("Random Forest Classifier:")
    print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, rfc_predict)))
    print('Training set score: {:.4f}'.format(cv.score(X_train, y_train)))
    print('Test set score: {:.4f}'.format(cv.score(X_test, y_test)))
    print(classification_report(y_test, rfc_predict))

    print('*************************************************************************')
execute()