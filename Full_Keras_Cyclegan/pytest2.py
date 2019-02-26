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

input_shape = (1, 600 ,1)
df = 64


def build_domain_predictor_keras(outname=None):
    
    dompred_0 = Input(shape=input_shape)
    dompred_1 = Flatten()(dompred_0)

    # Fully-connected layers
    dompred_2 = Dense(1024)(dompred_1)
    dompred_2 = LeakyReLU(alpha=0.2)(dompred_2)
    dompred_3 = Dense(1024)(dompred_2)
    dompred_3 = LeakyReLU(alpha=0.2)(dompred_3)

    # output layer
    dompred_out = Dense(2, name=outname)(dompred_3)
    #dompred_out = Reshape((1, 2), name=outname)(dompred_out)

    return Model(dompred_0, dompred_out)

dompred = build_domain_predictor_keras(outname='Dom_Pred')

dompred.summary()


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
    #dis_1 = d_conv2d(dis_0, filters=df)
    #dis_2 = d_conv2d(dis_1, filters=df*2)
    # Flatten
    dis_2 = Flatten()(dis_0)
    # FC layers
    dis_3 = Dense(512)(dis_2)
    dis_3 = LeakyReLU(alpha=0.2)(dis_3)
    dis_4 = Dense(512)(dis_3)
    dis_4 = LeakyReLU(alpha=0.2)(dis_4)

    # output layer
    dis_out = Dense(1, name=outname)(dis_4)

    return Model(dis_0, dis_out)

disc = build_discriminator_keras(outname='Disc')

disc.summary()