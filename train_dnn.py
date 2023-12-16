import numpy as np
import tensorflow as tf
import keras

from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def execute():
    x = pd.read_csv("data/existing/feature_VBL-VA001.csv", header=None)
    y = pd.read_csv("data/existing/label_VBL-VA001.csv", header=None)
    y = pd.Series.ravel(y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )
    print('*************************************************************************')
    print("Dense Neural Network Classifier")
    print("Shape of Train Data : {}".format(X_train.shape))
    print("Shape of Test Data : {}".format(X_test.shape))
    model1 = Sequential()
    model1.add(Dense(128, input_shape=(27,), activation='relu'))
    model1.add(Dense(128))
    model1.add(Dense(64))
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model1.summary)
    print(X_train.shape)
    model1.fit(X_train, Y_train, epochs=10, batch_size=256)
    y_pred_proba = model1.predict(X_test)
    y_pred = tf.argmax(y_pred_proba, axis=1)
    # Calculate and print accuracy
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    conf_matrix = confusion_matrix(Y_test, y_pred_proba)
    print("\nConfusion Matrix:\n", conf_matrix)
    # Display classification report
    class_report = classification_report(Y_test, y_pred)
    print("Classification Report:\n", class_report)
    print('Testing Set Evaluation F1-Score=>', f1_score(Y_test, y_pred_proba, average='macro'))
    print('*************************************************************************')


execute()

