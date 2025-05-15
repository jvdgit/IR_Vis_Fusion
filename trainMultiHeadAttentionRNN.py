#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:26:59 2017

@author: javed
"""

import numpy as np
from tensorflow import keras
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import  Conv1D, LSTM, GRU, Bidirectional, Flatten, GlobalAveragePooling1D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy.io
# import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from keras_multi_head import MultiHead
from keras_multi_head import MultiHeadAttention
from tensorflow.keras import layers, models

#from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.layers import BatchNormalization
#from keras_self_attention import SeqSelfAttention
#from temporal_pooling import TemporalAveragePooling2D
# fix random seed for reproducibility
np.random.seed(7)

mat = scipy.io.loadmat('fused_mvit_mhsa.mat')
# mat = scipy.io.loadmat('color16x4_mvit.mat')
sname =          'e4r5.mat'
h = 4

# final_weights_path = 'sa23.hdf5'
# 
# pname =             'Features_jpl_ofi8_1_ResNet50_fc1_rnn.mat'
final_weights_path = 'wse.hdf5'
# final_weights_path = 'weights_color16x4_mvit_mhsa.hdf5'


x_train = mat['x1']
y_train = mat['y1']
x_test = mat['x2']
y_test = mat['y2']

y1 = mat['y1']
y2 = mat['y2']


# data_dim = 512*2
data_dim = 256*2
#data_dim = 2048
timesteps = 1

num_classes = 12
nb_epoch = 22
model_path = 'models'

x_train = x_train.reshape(x_train.shape[0], timesteps,data_dim)
x_test = x_test.reshape(x_test.shape[0], timesteps, data_dim)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

start_time = time.time()


model = Sequential()
model.add(MultiHeadAttention(head_num=h,name='Multi-Head', input_shape=(timesteps, data_dim)))
del.add(keras.layers.Flatten(name='Flatten'))
model.add(Dense(64, activation='relu'))  # return a single vector of dimension 32
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

# model.load_weights(final_weights_path)

model.compile(loss='categorical_crossentropy',
                # optimizer='RMSprop',
                optimizer = RMSprop(lr=0.0001),
              metrics=['accuracy'])

# print(model.summary())

#final_weights_path = os.path.join(os.path.abspath(model_path), 'weights_skeleton_imp_joints_rotated_60.h5')

callbacks_list = [
     ModelCheckpoint(final_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True)
]

history=model.fit(x_train, y_train,
          batch_size=8, epochs=nb_epoch,
          callbacks=callbacks_list,
          validation_data=(x_test, y_test))

model.load_weights(final_weights_path)

y_pred = model.predict(x_test)
scipy.io.savemat(sname, mdict={'y_pred': y_pred})

get_last_layer_output = K.function([model.layers[0].input],
                                  [model.layers[-3].output])
# #
trainFeatures = get_last_layer_output([x_train,1])[0]
testFeatures = get_last_layer_output([x_test,0])[0]
#
scipy.io.savemat(pname,mdict={'trainFeatures': trainFeatures, 'testFeatures': testFeatures, 'y1':y1, 'y2':y2})
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))