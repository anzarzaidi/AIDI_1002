
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

def execute():
    global confusion_matrix
    x = pd.read_csv("data/existing/feature_VBL-VA001.csv", header=None)
    y = pd.read_csv("data/existing/label_VBL-VA001.csv", header=None)
    y = pd.Series.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )
    print('*************************************************************************')
    print("RandomForest Classifier")
    print("Shape of Train Data : {}".format(X_train.shape))
    print("Shape of Test Data : {}".format(X_test.shape))
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(criterion='entropy', random_state=42)
    rfc.fit(X_train, y_train)
    train_accuracy = rfc.score(X_train, y_train)
    test_accuracy = rfc.score(X_test, y_test)
    rfc_pred_train = rfc.predict(X_train)
    print(f"Train acc: {train_accuracy}")
    print(f"Test acc: {test_accuracy}")
    print('Training Set Evaluation F1-Score=>', f1_score(y_train, rfc_pred_train, average='micro'))
    rfc_pred_test = rfc.predict(X_test)
    print('Testing Set Evaluation F1-Score=>', f1_score(y_test, rfc_pred_test, average='micro'))
    confusion_matrix = confusion_matrix(y_test, rfc_pred_test)
    print(confusion_matrix)
    print(classification_report(y_test, rfc_pred_test))
    cv = GridSearchCV(rfc, parameters, cv=5)
    cv.fit(X_train, y_train)

    print_results(cv)
    rfc_predict = cv.predict(X_test)
    print("Random Forest Classifier:")
    print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, rfc_predict)))
    print('Training set score: {:.4f}'.format(cv.score(X_train, y_train)))
    print('Test set score: {:.4f}'.format(cv.score(X_test, y_test)))

    print(classification_report(y_test, rfc_predict))
    print('*************************************************************************')




def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

parameters = {
    'n_estimators': [5, 50, 250],
    'max_depth': [2, 4, 8, 16, 32, None]
}





execute()