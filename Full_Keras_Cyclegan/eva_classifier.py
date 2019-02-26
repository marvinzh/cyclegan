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

# Load saved models
model_path = './models/1811300338/model.h5'
model = load_model(model_path)

# Preparing test data
input_folder_test = './exp/ivectors_swbd_train/'
test_data = data_prep.datalist_load(foldername=input_folder_test, train=0)
test_data = test_data[:699]
test_data = np.array(test_data)
test_data = test_data.reshape(-1, 1, IVEC_DIM, 1)

test_labels = []

for i in range(0, len(test_data)):
    test_labels.append([1, 0])

test_labels = np.array(test_labels)

# Evaluation
score = model.evaluate(test_data, test_labels)
test_pred = model.predict_classes(test_data)

print (score)
print (test_pred)

print ("Evaluation is finished.")
