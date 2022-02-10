# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 11:03:24 2020

@author: bsr
"""

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers.core import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


print("X_train original shape", X_train.shape)
print("Y_train original shape", Y_train.shape)
print("X_test original shape", X_test.shape)
print("Y_test original shape", Y_test.shape)

plt.imshow(X_train[0], cmap="gray")
plt.title('Class' + str(Y_train[0]))

features_train=X_train.reshape(X_train.shape[0], 28, 28, 1)
features_test=X_test.reshape(X_test.shape[0], 28, 28, 1)

features_train= features_train.astype('float32')
features_test= features_test.astype('float32')

features_train/=255
features_test/=255

targets_train=np_utils.to_categorical(Y_train, 10)
targets_test=np_utils.to_categorical(Y_test, 10)

model = Sequential()
model.add(Conv2D(32, (3,3), activation= 'relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation= 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
model.fit(features_train,targets_train,epochs=2,verbose=1)
score = model.evaluate(features_test, targets_test)
print("Test accuracy: %.2f" %score[1])
