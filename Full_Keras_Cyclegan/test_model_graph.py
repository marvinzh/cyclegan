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


class CycleGAN():
    """The CycleGAN module."""
    def __init__(self):
        self.input_shape = (1, model_cyclegan.IVEC_DIM, 1)
        #self.num_fake_inputs = 0

        # Load parameters setting
        Parameters = para_setting.set_parameters()
        lambda_cycle = Parameters['lambda_cycle']
        gamma_ide = Parameters['gamma_ide']
        beta_gp = Parameters['beta_gpwgan']
        self.batchsize = Parameters['batch_size']
        self.pool_size = Parameters['pool_size']
        self.condition = Parameters['condition']

        self.fake_ivec_pool_A = np.zeros((self.pool_size, 1, model_cyclegan.IVEC_DIM, 1))
        self.fake_ivec_pool_B = np.zeros((self.pool_size, 1, model_cyclegan.IVEC_DIM, 1))

        # Initialize Generators
        self.G_AB = model_cyclegan.build_generator_keras(outname='G_AB_Out') # model 1
        self.G_BA = model_cyclegan.build_generator_keras(outname='G_BA_Out') # model 2
        ivec_a = Input(shape=self.input_shape, name='ivec_a')
        ivec_b = Input(shape=self.input_shape, name='ivec_b')

        # Generate faked i-vectors
        # Here I separate the generation of fake_ivec for G and D.
        # Because fake_ivec for D requires redefinition by as Input while G requires not.
        fake_ivec_b = self.G_AB(ivec_a)
        fake_ivec_b_for_D = self.G_AB(ivec_a)
        fake_ivec_b_for_D = Input(shape=self.input_shape, name='fake_ivec_b_for_D')
        fake_ivec_a = self.G_BA(ivec_b)
        fake_ivec_a_for_D = self.G_BA(ivec_b)
        fake_ivec_a_for_D = Input(shape=self.input_shape, name='fake_ivec_a_for_D')
        # Generate cycle i-vectors
        cycle_ivec_a = self.G_BA(fake_ivec_b)
        cycle_ivec_b = self.G_AB(fake_ivec_a)
        # Generate identity i-cvectors
        ide_ivec_b = self.G_AB(ivec_b)
        ide_ivec_a = self.G_BA(ivec_a)

        # Initialize the discriminators
        self.D_B = model_cyclegan.build_discriminator_keras(outname='D_B_Out') # model 3
        self.D_A = model_cyclegan.build_discriminator_keras(outname='D_A_Out') # model 4

        # Calculate the validity from Discriminators
        validity_b = self.D_B(fake_ivec_b_for_D)
        validity_b_for_G = self.D_B(fake_ivec_b)
        isreal_b = self.D_B(ivec_b)
        validity_a = self.D_A(fake_ivec_a_for_D)
        validity_a_for_G = self.D_A(fake_ivec_a)
        isreal_a = self.D_A(ivec_a)
        # Construct weighted average between real and fake images
        interpolateD_B = losses_keras.RandomWeightedAverage()([ivec_b, fake_ivec_b_for_D]) # model 5
        interpolateD_A = losses_keras.RandomWeightedAverage()([ivec_a, fake_ivec_a_for_D])
        # Determine validity of weighted sample
        validity_interpolateD_B = self.D_B(interpolateD_B)
        validity_interpolateD_A = self.D_A(interpolateD_A)

        partial_gp_loss_b = partial(
            losses_keras.gradient_penalty_loss,
            averaged_samples=interpolateD_B)
        partial_gp_loss_b.__name__ = 'gradient_penalty'  # Keras requires function names
        partial_gp_loss_a = partial(
            losses_keras.gradient_penalty_loss,
            averaged_samples=interpolateD_A)
        partial_gp_loss_a.__name__ = 'gradient_penalty'  # Keras requires function names

        # Compile discriminators(also called "Critics")
        #optimizer_d = SGD(lr=Parameters['base_lr'])
        optimizer_d = RMSprop(lr=Parameters['base_lr'])
        #optimizer_d = Adam(lr=Parameters['base_lr'], beta_1=0.5, beta_2=0.9)

        self.D_B_model = Model(
            inputs=[fake_ivec_b_for_D, ivec_b],
            outputs=[isreal_b, validity_b, validity_interpolateD_B]
        )
        self.D_B_model.compile(
            loss=[losses_keras.wasserstein_loss, 
                losses_keras.wasserstein_loss, 
                partial_gp_loss_b], 
            loss_weights=[1, 1, beta_gp], 
            optimizer=optimizer_d)
        
        self.D_A_model = Model(
            inputs=[fake_ivec_a_for_D, ivec_a],
            outputs=[isreal_a, validity_a, validity_interpolateD_A]
        )
        self.D_A_model.compile(
            loss=[losses_keras.wasserstein_loss, 
                losses_keras.wasserstein_loss, 
                partial_gp_loss_a], 
            loss_weights=[1, 1, beta_gp], 
            optimizer=optimizer_d)

        # Combine two generators for training.

        #optimizer_g = SGD(lr=Parameters['base_lr'])
        optimizer_g = RMSprop(lr=Parameters['base_lr'])
        #optimizer_g = Adam(lr=Parameters['base_lr'], beta_1=0.5, beta_2=0.9)

        self.combined_Gs = Model(
            inputs=[ivec_a, ivec_b], 
            outputs=[validity_a_for_G, validity_b_for_G, 
                cycle_ivec_a, cycle_ivec_b, 
                ide_ivec_a, ide_ivec_b]
        ) # include model 8, 9 ,10 and 11

        self.combined_Gs.compile(
            loss=['mse', 'mse',
                'mae', 'mae',
                'mae', 'mae'],
            loss_weights=[1, 1, 
                lambda_cycle, lambda_cycle,
                gamma_ide, gamma_ide], 
            optimizer=optimizer_g)

        #self.D_A_model.summary()
        #self.D_B_model.summary()
        #self.combined_Gs.summary()
        
        model_path = './models/TEST'
        model_Gs_path = model_path + '/combined_Gs_0.h5'
        self.combined_Gs.save(model_Gs_path)

        model_DB_path = model_path + '/D_B_model_0.h5'
        self.D_B_model.save(model_DB_path)

        model_DA_path = model_path + '/D_A_model_0.h5'
        self.D_A_model.save(model_DA_path)

        self.combined_Gs.get_layer('model_3').summary()
        
    '''
    def train(self):
        self.combined_Gs.get_layer('model_1').summary()
        print (self.combined_Gs.get_layer('model_1').get_output_at(0))
        print (self.combined_Gs.get_layer('model_1').get_output_at(1))
        print (self.combined_Gs.get_layer('model_1').get_output_at(2))
        print (self.combined_Gs.get_layer('model_1').get_output_at(3))

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


        OUT_1 = self.combined_Gs.get_layer('model_1').predict(input_data_b)

        print (OUT_1.shape)
    '''


gan = CycleGAN()

Gs = load_model('./models/TEST/combined_Gs_0.h5')
Gs.summary()

#Gs.get_layer('model_1').summary()
print (Gs.get_layer('model_1').get_output_at(0))
print (Gs.get_layer('model_1').get_output_at(1))
print (Gs.get_layer('model_1').get_output_at(2))
print (Gs.get_layer('model_1').get_output_at(3))

#Trans_A2B = Model(inputs=Gs.get_layer('ivec_a').input, outputs=Gs.get_layer('model_8').get_output_at(1)) 
