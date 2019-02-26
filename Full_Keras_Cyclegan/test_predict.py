#!/usr/bin/env python

import os
from datetime import datetime
import random
from functools import partial
import numpy as np

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import TensorBoard

import para_setting
import model_cyclegan, losses_keras
import data_prep


Gs = load_model('./models/1812041842_wiwaxia1/combined_Gs_20.h5')
Gs.summary()

# Choose evaluation condition and enroll("_train") or test dataset folder
input_folder_enroll = './exp/C5/ivectors_sre10_train/'
input_folder_test = './exp/C5/ivectors_sre10_test/'

for k in range(0, 2):
    if k == 0:
        input_data_b, input_label_b = data_prep.datalist_load(foldername=input_folder_test, train=0, use_label=True)
    elif k == 1:
        input_data_b, input_label_b = data_prep.datalist_load(foldername=input_folder_enroll, train=0, use_label=True)
        
input_data_b = np.array(input_data_b[0])
input_data_b = input_data_b.reshape((-1, 1, model_cyclegan.IVEC_DIM, 1))

OUT_1 = Gs.get_layer('model_1').predict(input_data_b)

print (OUT_1)
print (OUT_1.shape)

#Trans_A2B = Model(inputs=Gs.get_layer('ivec_a').input, outputs=Gs.get_layer('model_8').get_output_at(1)) 
