#!/usr/bin/env python

import os

import tensorflow as tf
#import tflearn
import keras

import datetime
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPool2D, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization, Dropout
from keras.utils import to_categorical

import data_prep
import kaldi_io
#from . import ResBlock_LReLU_keras
#from . import special_layers

# The number of samples per batch.
BATCH_SIZE = 1

# The height of each i-vector.
IVEC_HEIGHT = 1

# The length of each i-vector.
IVEC_DIM = 600

# The number of color channels per image.
IVEC_CHANNELS = 1

#Preparing train data
folder_input_swbd = './exp/ivectors_swbd_train/'  # swbd == a
folder_input_mixer = './exp/ivectors_mixer_train/'  #mixer == b
input_data_a = data_prep.datalist_load(foldername=folder_input_swbd, train=0)
input_data_b = data_prep.datalist_load(foldername=folder_input_mixer, train=0)
train_data = input_data_a + input_data_b
num_batch = len(train_data)
train_data = np.array(train_data)
train_data = train_data.reshape(-1, 1, IVEC_DIM, 1)

labels = []

for i in range(0, len(input_data_a)):
    labels.append([1, 0])

for i in range(0, len(input_data_b)):
    labels.append([0, 1])
    
labels = np.array(labels)
#labels = to_categorical(labels)

# Initialization
model = Sequential()

# conv layer 1
model.add(Conv2D(32, [1, 3], padding='same', strides=1, input_shape=(1, IVEC_DIM, 1)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.25))

# conv layer 2
model.add(Conv2D(64, [1, 3], padding='same', strides=1))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.25))

model.add(MaxPool2D(padding='same', strides=1))

model.add(Flatten())

model.add(Dense(512))
model.add(LeakyReLU(alpha=0.3))

model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

# Training
model.compile(optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_data, labels, epochs=1, batch_size=1)

'''
for i in range(0, num_batch):
    train_data_this_batch = tf.convert_to_tensor(train_data[i])
    train_data_this_batch = tf.reshape(train_data_this_batch, [1, 1, IVEC_DIM, 1])

    model.train_on_batch(train_data_this_batch, labels[i])
'''
print ("Training is finished.")

# Save trained models
model_time = datetime.datetime.now().strftime('%y%m%d%H%M')
os.makedirs('./models/' + model_time)
model_path = './models/' + model_time + '/model.h5'
model.save(model_path)

'''
# Load saved models
model = load_model(model_path)
'''

# Preparing test data
input_folder_test = './exp/C5/ivectors_sre10_test/'
test_data = data_prep.datalist_load(foldername=input_folder_test, train=0)
test_data = np.array(test_data)
test_data = test_data.reshape(-1, 1, IVEC_DIM, 1)

test_labels = []

#for i in range(0, len(test_data)):
#    test_labels.append([0, 1])

#test_labels = np.array(test_labels)

# Evaluation
#score = model.evaluate(test_data, test_labels)
test_pred = model.predict_classes(test_data)

#print (score)
print (test_pred)

print ("Evaluation is finished.")
