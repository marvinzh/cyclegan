#!/usr/bin/env python

import tensorflow as tf
import tflearn
import keras

import numpy as np

from keras.models import Model
from keras.layers import Conv2D, Concatenate, add
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers import InstanceNormalization

def Res_Block(layer_input, filters, strides=1, f_size=[1, 3]):
    res_1 = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
    res_1 = LeakyReLU(alpha=0.2)(res_1)
    res_1 = InstanceNormalization()(res_1)

    res_2 = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(res_1)
    res_2 = LeakyReLU(alpha=0.2)(res_2)
    res_2 = InstanceNormalization()(res_2)

    res_out = add([res_2, layer_input])
    return res_out

'''
g_0 = Input(shape=(1, 600, 1))    
inlayer = Conv2D(32, [1, 3], padding='same', strides=[1, 2])(g_0)
outlayer = Res_Block(inlayer, 32)

print (inlayer)
print (outlayer)
'''