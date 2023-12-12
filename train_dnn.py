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

# load data hasil ekstraksi fitur fft
x = pd.read_csv("data/extracted/feature_VBL-VA001.csv", header=None)

# load label
y = pd.read_csv("data/extracted/label_VBL-VA001.csv", header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)
print("Shape of Train Data : {}".format(X_train.shape))
print("Shape of Test Data : {}".format(X_test.shape))

model1 = Sequential()
model1.add(Dense(128, input_shape=(27,), activation = 'relu'))
model1.add(Dense(128))
model1.add(Dense(64))
model1.add(Dense(1,  activation = 'sigmoid'))
model1.compile(optimizer='rmsprop', loss ='categorical_crossentropy', metrics=['accuracy'])
print(model1.summary)
print(X_train.shape)
model1.fit(X_train, Y_train, epochs = 10, batch_size = 256)

