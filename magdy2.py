

from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Optimizer
from keras.utils import np_utils
import tensorflow as tf
from keras.utils import np_utils

############################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(y_test.shape))


############################################
X_train =X_train.reshape(60000,784)

X_test=X_test.reshape(10000,784)
###########################
X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255
############################################
y_train=keras.utils.np_utils.to_categorical(y_train,10)

y_test=keras.utils.np_utils.to_categorical(y_test,10)

#############################################
model = keras.Sequential([
    keras.layers.Dense(512,input_dim=784 ,activation=tf.nn.relu),
    keras.layers.Dense(256,activation=tf.nn.relu),
    keras.layers.Dense(124 ,activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
###############################################

model.compile (loss=keras.losses.CategoricalCrossentropy(),
               metrics=['accuracy'])

model.fit(X_train, y_train,epochs=5, batch_size=10)


################################################
score= model.evaluate(X_test, y_test, verbose=0)
print ('test loss ',score[0])
print ('test accuaracy ',score[1])









