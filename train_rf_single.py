# script to train VBL-VA001

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

#load data
x = pd.read_csv("data/feature_VBL-VA001.csv", header=None)
# load label
y = pd.read_csv("data/label_VBL-VA001.csv", header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)


print("Shape of Train Data : {}".format(X_train.shape))
print("Shape of Test Data : {}".format(X_test.shape))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion = 'entropy', random_state = 42)
rfc.fit(X_train, y_train)

train_accuracy = rfc.score(X_train, y_train)
# Compute accuracy on the test set
test_accuracy = rfc.score(X_test, y_test)
rfc_pred_train = rfc.predict(X_train)
print(f"Train acc: {train_accuracy}")

print(f"Test acc: {test_accuracy}")
# Evaluating on Training set
print('Training Set Evaluation F1-Score=>',f1_score(y_train,rfc_pred_train,average='micro'))


rfc_pred_test = rfc.predict(X_test)
print('Testing Set Evaluation F1-Score=>',f1_score(y_test,rfc_pred_test,average='micro'))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, rfc_pred_test)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, rfc_pred_test))