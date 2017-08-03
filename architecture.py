
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.utils.np_utils import to_categorical

import numpy as np
import sys

# import dataset:
X_fname = 
y_fname = 
X_train = 
y_train = 
X_train = X_train.astype('float32')
y_train = to_categorical(y_train)

# params:
batch_size = 128
nb_epoch = 50

# setup info:
print 'X_train shape: ', X_train.shape # (n_sample, 1, 32, 32)
print 'y_train shape: ', y_train.shape # (n_sample, n_categories)
print '  img size: ', X_train.shape[2], X_train.shape[3]
print 'batch size: ', batch_size
print '  nb_epoch: ', nb_epoch

# model architecture:
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',input_shape=(1, X_train.shape[2], X_train.shape[3])))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))