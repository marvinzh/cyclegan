#!/usr/bin/env python

#import tensorflow as tf
#import tflearn
import keras

import numpy as np

#from keras.models import Sequential
#from keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, MaxPool2D, Activation
#from keras.layers.advanced_activations import LeakyReLU
#from keras.layers import BatchNormalization, Dropout

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

'''
folder_input_swbd = './exp/ivectors_swbd_train/'  # swbd == a
folder_input_mixer = './exp/ivectors_mixer_train/'  #mixer == b
input_data_a = data_prep.datalist_load(foldername=folder_input_swbd, train=0)
input_data_b = data_prep.datalist_load(foldername=folder_input_mixer, train=0)
train_data = input_data_a + input_data_b

train_data = np.array(train_data)
labels = []

for i in range(0, len(input_data_a)):
    labels.append([0, 1])

for i in range(0, len(input_data_b)):
    labels.append([1, 0])


labels = np.array(labels)

print (labels[0].shape)
print (len(train_data))
print (len(labels))

# Preparing test data
input_folder_test = './exp/C5/ivectors_sre10_test/'
test_data = data_prep.datalist_load(foldername=input_folder_test, train=0)
test_data = np.array(test_data)

test_labels = []

for i in range(0, len(test_data)):
    test_labels.append([1, 0])

test_labels = np.array(test_labels)

print (test_data)
print (test_data.shape)
print (test_labels.shape)
'''


def build_generator_keras(outname=None):
    # Define layers
    def conv2d(layer_input, filters, strides=1, f_size=[1, 3], dropout_rate=0, IN=True, name=None):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', kernel_initializer='he_uniform', bias_initializer='he_uniform', name=name)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if dropout_rate:
            d = Dropout(dropout_rate)(d)
        if IN:
            d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, filters, strides=1, f_size=[1, 3], dropout_rate=0, IN=True, name=None):
        """Layers used during upsampling"""
        #u = UpSampling2D(size=2)(layer_input)
        u = Conv2DTranspose(filters, kernel_size=f_size, strides=strides, padding='same', kernel_initializer='he_uniform', bias_initializer='he_uniform', name=name)(layer_input)
        u = LeakyReLU(alpha=0.2)(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        if IN:
            u = InstanceNormalization()(u)
        #u = Concatenate()([u, skip_input])
        return u

    # Build Generators
    g_0 = Input(shape=input_shape)
    # downsampling
    gc_1 = conv2d(g_0, filters=gf, dropout_rate=0.5, IN=False)
    #gc_1 = MaxPool2D((1, 2), padding='same')(gc_1)
    gc_2 = conv2d(gc_1, filters=gf*2, dropout_rate=0.5)
    #gc_2 = MaxPool2D((1, 2), padding='same')(gc_2)
    gc_3 = conv2d(gc_2, filters=gf*4, dropout_rate=0.5)

    # res_net with 6 blocks
    g_res = Res_Block(gc_3, gf*4)
    g_res = Res_Block(g_res, gf*4)
    g_res = Res_Block(g_res, gf*4)
    g_res = Res_Block(g_res, gf*4)
    g_res = Res_Block(g_res, gf*4)
    g_res = Res_Block(g_res, gf*4)

    # upsampling
    # deconv layer 1
    gu_1 = deconv2d(g_res, filters=gf*2, dropout_rate=0.5)
    gu_2 = deconv2d(gu_1, filters=gf, dropout_rate=0.5)

    #output layer
    g_out = Conv2D(1, [1, 3], padding='same', strides=1, activation='tanh', name=outname)(gu_2)
    #g_out = Reshape((1, IVEC_DIM, 1))(g_out)

    return Model(g_0, g_out)

G_AB = build_generator_keras()
