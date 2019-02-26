#!/usr/bin/env python

import tensorflow as tf
import tflearn
import keras

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, UpSampling2D, Conv2DTranspose, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Reshape, Dropout, Concatenate, BatchNormalization
from keras_contrib.layers import InstanceNormalization

from res_block import Res_Block
import gradient_reversal_keras


# The height of each i-vector.
IVEC_HEIGHT = 1

# The length of each i-vector.
IVEC_DIM = 600

# The number of color channels per image.
IVEC_CHANNELS = 1

gf = 32
df = 64
input_shape = (1, IVEC_DIM, 1)

'''
def build_generator_keras(outname=None):
    # Define layers
    def conv2d(layer_input, filters, strides=1, f_size=[1, 3], dropout_rate=0, name=None):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', kernel_initializer='he_uniform', bias_initializer='he_uniform', name=name)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if dropout_rate:
            d = Dropout(dropout_rate)(d)
        d = InstanceNormalization()(d)
        return d

    # Build Generators
    g_0 = Input(shape=input_shape)

    gc_1 = conv2d(g_0, filters=gf*2, dropout_rate=0.4)
    gc_1 = MaxPool2D((1, 2), padding='same')(gc_1)
    gc_2 = conv2d(gc_1, filters=gf*4, dropout_rate=0.4)
    gc_2 = MaxPool2D((1, 2), padding='same')(gc_2)
    gc_3 = conv2d(gc_2, filters=gf*8, dropout_rate=0.4)
    # Flatten
    gc_3 = Flatten()(gc_3)
    # FC layers
    gc_4 = Dense(512)(gc_3)
    gc_4 = LeakyReLU(alpha=0.2)(gc_4)
    gc_5 = Dense(512)(gc_4)
    gc_5 = LeakyReLU(alpha=0.2)(gc_5)

    #output layer
    #g_out = Conv2D(1, [1, 3], padding='same', strides=1, name=outname)(gc_3)
    g_out = Dense(IVEC_DIM, activation='tanh', name=outname)(gc_5)
    g_out = Reshape((1, IVEC_DIM, 1))(g_out)

    return Model(g_0, g_out)
'''

def build_generator_keras(outname=None):
    # Define layers
    def conv2d(layer_input, filters, strides=1, f_size=[1, 3], dropout_rate=0, name=None):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', kernel_initializer='he_uniform', bias_initializer='he_uniform', name=name)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if dropout_rate:
            d = Dropout(dropout_rate)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, filters, strides=1, f_size=[1, 3], dropout_rate=0, name=None):
        """Layers used during upsampling"""
        #u = UpSampling2D(size=2)(layer_input)
        u = Conv2DTranspose(filters, kernel_size=f_size, strides=strides, padding='same', kernel_initializer='he_uniform', bias_initializer='he_uniform', name=name)(layer_input)
        u = LeakyReLU(alpha=0.2)(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        #u = Concatenate()([u, skip_input])
        return u

    # Build Generators
    g_0 = Input(shape=input_shape)
    # downsampling
    gc_1 = conv2d(g_0, filters=gf, dropout_rate=0.5)
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


def build_discriminator_keras(outname=None):
    # Define layers
    def d_conv2d(layer_input, filters, strides=1, f_size=[1, 3], name=None):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        return d

    #Build Discriminators
    #inputdisc = Reshape((1, IVEC_DIM, 1))(inputdisc)
    dis_0 = Input(shape=input_shape)
    # Conv layers
    dis_1 = d_conv2d(dis_0, filters=df)
    dis_2 = d_conv2d(dis_1, filters=df*2)
    # Flatten
    dis_2 = Flatten()(dis_2)
    # FC layers
    dis_3 = Dense(512)(dis_2)
    dis_3 = LeakyReLU(alpha=0.2)(dis_3)
    dis_4 = Dense(512)(dis_3)
    dis_4 = LeakyReLU(alpha=0.2)(dis_4)

    # output layer
    dis_out = Dense(1, name=outname)(dis_4)

    return Model(dis_0, dis_out)

def build_domain_predictor_keras(outname=None):
    # Gradient Reversal Layer
    Flip = gradient_reversal_keras.GradientReversal(hp_lambda=0.5)

    dompred_0 = Input(shape=input_shape)
    dompred_1 = Flatten()(dompred_0)
    dompred_1 = Flip(dompred_1)

    # Fully-connected layers
    dompred_2 = Dense(1024)(dompred_1)
    dompred_2 = LeakyReLU(alpha=0.2)(dompred_2)
    dompred_3 = Dense(1024)(dompred_2)
    dompred_3 = LeakyReLU(alpha=0.2)(dompred_3)

    # output layer
    dompred_out = Dense(2, name=outname)(dompred_3)
    #dompred_out = Reshape((1, 2), name=outname)(dompred_out)

    return Model(dompred_0, dompred_out)

    '''
    # Define layers
    def domp_conv2d(layer_input, filters, strides=2, f_size=[1, 3]):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        return d

    # Conv layers
    dompred_1 = domp_conv2d(dompred_0, filters=df, strides=1)
    dompred_2 = domp_conv2d(dompred_1, filters=df*2)
    # Flatten
    dompred_2 = Flatten()(dompred_2)
    # FC layers
    dompred_3 = Dense(512)(dompred_2)
    dompred_3 = LeakyReLU(alpha=0.2)(dompred_3)
    dompred_4 = Dense(512)(dompred_3)
    dompred_4 = LeakyReLU(alpha=0.2)(dompred_4)
    
    # output layer
    # "tf.nn.sparse_softmax_cross_entropy_with_logits" will perform softmax.
    dompred_out = Dense(2)(dompred_4)
    dompred_out = tf.reshape(dompred_out, [1, 2])
    '''
