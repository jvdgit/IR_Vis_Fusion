#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 21:16:37 2017

@author: javed
"""

import numpy as np
from tensorflow import keras
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import  MaxPool1D, Conv1D, LSTM, GRU, Bidirectional, Flatten, GlobalAveragePooling1D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy.io
# import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import SGD, RMSprop
#from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.layers import BatchNormalization
#from keras_self_attention import SeqSelfAttention
#from temporal_pooling import TemporalAveragePooling2D
# fix random seed for reproducibility
np.random.seed(7)
# fix random seed for reproducibility
np.random.seed(7)
from keras_multi_head import MultiHeadAttention
from tensorflow.keras import backend as K
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.backend import eager_learning_phase_scope

mat = scipy.io.loadmat('fused_mvit_1dcnn.mat')


final_weights_path = 'sa.h5'


x_train = mat['x1']
y_train = mat['y1']
x_test = mat['x2']
y_test = mat['y2']

y1 = mat['y1']
y2 = mat['y2']

data_dim = 256*2
timesteps = 1
num_classes = 12
nb_epoch = 15
model_path = 'models'
ksize=1

x_train = x_train.reshape(x_train.shape[0], timesteps,data_dim)
x_test = x_test.reshape(x_test.shape[0], timesteps, data_dim)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = Sequential()                                               
model.add(Conv1D(64,ksize,input_shape=(timesteps,data_dim), activation='relu'))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
               # optimizer='adam',
                optimizer=RMSprop(lr=0.0001),
              # optimizer = SGD(lr=0.001, momentum=0.9),
              metrics=['accuracy'])

# print(model.summary())

callbacks_list = [
     ModelCheckpoint(final_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True)
]

history=model.fit(x_train, y_train,
      batch_size=8, epochs=nb_epoch,
      callbacks=callbacks_list,
      validation_data=(x_test, y_test))


model.load_weights(final_weights_path)

get_last_layer_output = K.function([model.layers[0].input],
                                  [model.layers[-2].output])


trainFeatures = np.zeros((len(x_train), 256))

i = 0

eager_learning_phase_scope(value=1)
for i in range(len(x_train)):
    trainFeatures[i,:] = get_last_layer_output(x_train[i:i+1,:,:])[0]
    if i%100 == 0:
        print(i)

testFeatures = np.zeros((len(x_test), 256))

eager_learning_phase_scope(value=0)

for i in range(len(x_test)):
        testFeatures[i,:] = get_last_layer_output(x_test[i:i+1,:,:])[0]
        if i%100 == 0:
            print(i)
            

scipy.io.savemat(sname, mdict={'trainFeatures': trainFeatures, 'testFeatures': testFeatures, 'y1':y1, 'y2':y2})
scipy.io.savemat(pname, mdict={'y_pred': y_pred})

print('\nFeatures extracted successfully!!')