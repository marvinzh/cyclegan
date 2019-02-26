#!/usr/bin/env python

from datetime import datetime
import json
import os
import random
import numpy as np
from functools import partial

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD, Adam, RMSprop

import para_setting
import losses, model_cyclegan, losses_keras
import data_prep


input_shape=(1, model_cyclegan.IVEC_DIM, 1)
# Load parameters setting
Parameters = para_setting.set_parameters()
lambda_cycle = Parameters['lambda_cycle']
gamma_ide = Parameters['gamma_ide']
beta_gp = Parameters['beta_gpwgan']
# Initialize Generators
G_AB = model_cyclegan.build_generator_keras() # input_1
G_BA = model_cyclegan.build_generator_keras() # input_2

folder_input_swbd = './exp/ivectors_swbd_train/'  # swbd == a
folder_input_mixer = './exp/ivectors_mixer_train/'  #mixer == b
ivec_a = data_prep.datalist_load(foldername=folder_input_swbd, train=0)
ivec_a = tf.convert_to_tensor(ivec_a[10])
#ivec_a = np.array(ivec_a)
ivec_a = tf.reshape(ivec_a, [-1, 1, model_cyclegan.IVEC_DIM, 1])
ivec_a = Input(shape=input_shape) # input_3
ivec_b = data_prep.datalist_load(foldername=folder_input_mixer, train=0)
ivec_b = tf.convert_to_tensor(ivec_b[10])
#ivec_b = np.array(ivec_b)
ivec_b = tf.reshape(ivec_b, [-1, 1, model_cyclegan.IVEC_DIM, 1])
ivec_b = Input(shape=input_shape) # input_4

# Generate faked i-vectors
# Here I separate the generation of fake_ivec for G and D.
# Because fake_ivec for D requires redefinition by as Input while G requires not.
fake_ivec_b = G_AB(ivec_a)
fake_ivec_b_for_D = G_AB(ivec_a)
fake_ivec_b_for_D = Input(shape=input_shape) # input_5
fake_ivec_a = G_BA(ivec_b)
fake_ivec_a_for_D = G_BA(ivec_b)
fake_ivec_a_for_D = Input(shape=input_shape) # input_6
# Generate cycle i-vectors
cycle_ivec_a = G_BA(fake_ivec_b)
cycle_ivec_b = G_AB(fake_ivec_a)
# Generate identity i-cvectors
ide_ivec_b = G_AB(ivec_b)
ide_ivec_a = G_BA(ivec_a)
# Initialize the discriminators
D_B = model_cyclegan.build_discriminator_keras() # input_7
D_A = model_cyclegan.build_discriminator_keras() # input_8

G_AB.summary()
G_BA.summary()
D_B.summary()
D_A.summary()

# Calculate the validity from Discriminators
#prob_real_b_is_real = D_B(fake_ivec_b)
#prob_real_a_is_real = D_A(fake_ivec_a)
validity_b = D_B(fake_ivec_b_for_D)
validity_b_for_G = D_B(fake_ivec_b)
isreal_b = D_B(ivec_b)
validity_a = D_A(fake_ivec_a_for_D)
validity_a_for_G = D_A(fake_ivec_a)
isreal_a = D_A(ivec_a)
# Construct weighted average between real and fake images
interpolated_b = losses_keras.RandomWeightedAverage()([ivec_b, fake_ivec_b_for_D])
interpolated_a = losses_keras.RandomWeightedAverage()([ivec_a, fake_ivec_a_for_D])
# Determine validity of weighted sample
validity_interpolated_b = D_B(interpolated_b)
validity_interpolated_a = D_A(interpolated_a)

partial_gp_loss_b = partial(
    losses_keras.gradient_penalty_loss, 
    averaged_samples=interpolated_b)
partial_gp_loss_b.__name__ = 'gradient_penalty' # Keras requires function names
partial_gp_loss_a = partial(
    losses_keras.gradient_penalty_loss, 
    averaged_samples=interpolated_a)
partial_gp_loss_a.__name__ = 'gradient_penalty' # Keras requires function names

# Compile discriminators(also called "Critics")
#optimizer_d = SGD(lr=Parameters['base_lr'])
optimizer_d = Adam(0.0001, beta_1=0.5, beta_2=0.9)

print (isreal_b)
print (validity_b)
print (validity_interpolated_b)

D_B_model = Model(
    inputs=[fake_ivec_b_for_D, ivec_b],
    outputs=[isreal_b, validity_b, validity_interpolated_b]
)
D_B_model.compile(
    loss=[losses_keras.wasserstein_loss, 
        losses_keras.wasserstein_loss, 
        partial_gp_loss_b], 
    loss_weights=[1, 1, beta_gp], 
    optimizer=optimizer_d)

D_A_model = Model(
    inputs=[fake_ivec_a_for_D, ivec_a],
    outputs=[isreal_a, validity_a, validity_interpolated_a]
)
D_A_model.compile(
    loss=[losses_keras.wasserstein_loss, 
        losses_keras.wasserstein_loss, 
        partial_gp_loss_a], 
    loss_weights=[1, 1, beta_gp], 
    optimizer=optimizer_d)


# Combine two generators for training.
combined_Gs = Model(
    inputs=[ivec_a, ivec_b], 
    outputs=[validity_a_for_G, validity_b_for_G, 
        cycle_ivec_a, cycle_ivec_b, 
        ide_ivec_a, ide_ivec_b]
)

#optimizer_g = SGD(lr=Parameters['base_lr'])
optimizer_g = Adam(0.0001, beta_1=0.5, beta_2=0.9)

combined_Gs.compile(
    loss=[losses_keras.wasserstein_loss, losses_keras.wasserstein_loss,
        'mae', 'mae',
        'mae', 'mae'],
    loss_weights=[1, 1, 
        lambda_cycle, lambda_cycle,
        gamma_ide, gamma_ide],
    optimizer=optimizer_g)

print ('Finished.')
