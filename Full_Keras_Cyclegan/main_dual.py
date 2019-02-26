#!/usr/bin/env python

import os
from datetime import datetime
from functools import partial
import numpy as np

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
        #beta_gp = Parameters['beta_gpwgan']
        self.batchsize = Parameters['batch_size']
        self.pool_size = Parameters['pool_size']

        #self.fake_ivec_pool_A = np.zeros((self.pool_size, 1, model_cyclegan.IVEC_DIM, 1))
        #self.fake_ivec_pool_B = np.zeros((self.pool_size, 1, model_cyclegan.IVEC_DIM, 1))

        # Initialize the discriminators
        self.D_B = model_cyclegan.build_discriminator_keras(outname='D_B_Out')  # model 1
        self.D_A = model_cyclegan.build_discriminator_keras(outname='D_A_Out')  # model 2

        #optimizer_d = SGD(lr=Parameters['base_lr'])
        optimizer_d = RMSprop(lr=Parameters['base_lr'])
        #optimizer_d = Adam(lr=Parameters['base_lr'], beta_1=0.5, beta_2=0.9)

        self.D_B.compile(
            loss=losses_keras.wasserstein_loss,
            optimizer=optimizer_d,
            metrics=['accuracy'])

        self.D_A.compile(
            loss=losses_keras.wasserstein_loss, 
            optimizer=optimizer_d,
            metrics=['accuracy'])

        # Initialize Generators
        self.G_AB = model_cyclegan.build_generator_keras(outname='G_AB_Out')  # model 3
        self.G_BA = model_cyclegan.build_generator_keras(outname='G_BA_Out')  # model 4

        # Freeze D_models during the training of combined model by setting trainable = False
        self.D_B.trainable = False
        self.D_A.trainable = False

        ivec_a = Input(shape=self.input_shape, name='ivec_a')
        ivec_b = Input(shape=self.input_shape, name='ivec_b')

        # Generate faked i-vectors
        fake_ivec_b = self.G_AB(ivec_a)
        fake_ivec_a = self.G_BA(ivec_b)
        # Generate cycle i-vectors
        cycle_ivec_a = self.G_BA(fake_ivec_b)
        cycle_ivec_b = self.G_AB(fake_ivec_a)
        # Generate identity i-cvectors
        ide_ivec_b = self.G_AB(ivec_b)
        ide_ivec_a = self.G_BA(ivec_a)

        # Generate validity of transformed i-vectors
        validity_b = self.D_B(fake_ivec_b)
        validity_a = self.D_A(fake_ivec_a)

        #optimizer_g = SGD(lr=Parameters['base_lr'])
        optimizer_g = RMSprop(lr=Parameters['base_lr'])
        #optimizer_g = Adam(lr=Parameters['base_lr'], beta_1=0.5, beta_2=0.9)

        self.combined_Gs = Model(
            inputs=[ivec_a, ivec_b], 
            outputs=[validity_a, validity_b, 
                cycle_ivec_a, cycle_ivec_b, 
                ide_ivec_a, ide_ivec_b]
        )

        self.combined_Gs.compile(
            loss=[losses_keras.wasserstein_loss, losses_keras.wasserstein_loss,
                'mae', 'mae',
                'mae', 'mae'],
            loss_weights=[1, 1, 
                lambda_cycle, lambda_cycle,
                gamma_ide, gamma_ide], 
            optimizer=optimizer_g)

        self.D_A.summary()
        self.D_B.summary()
        self.combined_Gs.summary()    

    '''
    def fake_ivec_pool(self, num_fakes, fake, fake_pool):
        """
        This function saves the generated ivector to corresponding
        pool of ivectors.

        It keeps on feeding the pool till it is full and then randomly
        selects an already stored ivectors and replace it with new one.
        """
        if num_fakes < self.pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self.pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                temp = temp.reshape(self.batchsize, 1, model_cyclegan.IVEC_DIM, 1)
                return temp
            else:
                return fake
    '''

    def train(self, epochs):
        print ("Start CycleGAN training.")
        start_time = datetime.now()
        batch_size = self.batchsize

        # Load training datasets from the folders
        folder_input_swbd = './exp/ivectors_swbd_train/'  # swbd == a
        folder_input_mixer = './exp/ivectors_mixer_train/'  #mixer == b
        ivec_A = data_prep.datalist_load(foldername=folder_input_swbd, train=1)
        ivec_A = np.array(ivec_A)
        ivec_A = ivec_A.reshape(-1, 1, model_cyclegan.IVEC_DIM, 1)
        ivec_B = data_prep.datalist_load(foldername=folder_input_mixer, train=1)
        ivec_B = np.array(ivec_B)
        ivec_B = ivec_B.reshape(-1, 1, model_cyclegan.IVEC_DIM, 1)

        max_data = len(ivec_A)
        num_batch = max_data // batch_size

        clip_value = 0.01

        # Ground truths for discriminators
        '''
        d_valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        '''
        d_valid = -np.ones((batch_size, 1))
        d_fake = np.ones((batch_size, 1))

        # Ground truths for generators
        #g_fake = -np.ones((batch_size, 1)) # -1

        #log_path = './graph'
        #callback = TensorBoard(log_path)
        #callback.set_model(model)
        #train_names = ['train_loss', 'train_mae']

        model_time = start_time.strftime('%y%m%d%H%M')
        model_path = './models/' + model_time
        os.makedirs(model_path)

        for epoch in range(epochs):
            for batch_i in range (0, num_batch):

                ivec_A_feed = ivec_A[batch_size * batch_i: batch_size * (batch_i + 1)]
                ivec_B_feed = ivec_B[batch_size * batch_i: batch_size * (batch_i + 1)]

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.G_AB.predict(ivec_A_feed)
                fake_A = self.G_BA.predict(ivec_B_feed)

                # Train the discriminators A and B
                # Label names here are a bit confusing. Labels order is [-1, 1]
                dB_loss_real = self.D_B.train_on_batch(ivec_B_feed, d_valid)
                dB_loss_fake = self.D_B.train_on_batch(fake_B, d_fake)

                dA_loss_real = self.D_A.train_on_batch(ivec_A_feed, d_valid)
                dA_loss_fake = self.D_A.train_on_batch(fake_A, d_fake)

                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                # Clip discriminator weights
                for d in [self.D_A, self.D_B]:
                    for l in d.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                        l.set_weights(weights)

                # Total disciminator loss
                D_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                G_loss = self.combined_Gs.train_on_batch([ivec_A_feed, ivec_B_feed],
                                                        [d_valid, d_valid,
                                                        ivec_A_feed, ivec_B_feed,
                                                        ivec_A_feed, ivec_B_feed])

                elapsed_time = datetime.now() - start_time

                # Plot the progress
                if batch_i % 100 == 0 or batch_i == num_batch:
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, cyc: %05f, id: %05f] time: %s " \
                                                                        % ( epoch + 1, epochs,
                                                                            batch_i, num_batch,
                                                                            D_loss[0], 100*D_loss[1],
                                                                            G_loss[0],
                                                                            np.mean(G_loss[1:3]),
                                                                            np.mean(G_loss[3:5]),
                                                                            np.mean(G_loss[5:6]),
                                                                            elapsed_time))


            # Save trained models
            model_num = str(epoch + 1)
            if int(model_num) % 5 == 0:
                model_Gs_path = model_path + '/combined_Gs_' + model_num + '.h5'
                self.combined_Gs.save(model_Gs_path)

                model_DB_path = model_path + '/D_B_' + model_num + '.h5'
                self.D_B.save(model_DB_path)

                model_DA_path = model_path + '/D_A_' + model_num + '.h5'
                self.D_A.save(model_DA_path)

            '''
            # Remove old models. Only keep the latest 3 epochs models
            rm_num = str(epoch - 2)
            os.system("rm -f %s" % (model_path + '/combined_Gs_' + rm_num + '.h5'))
            os.system("rm -f %s" % (model_path + '/D_B_' + rm_num + '.h5'))
            os.system("rm -f %s" % (model_path + '/D_A_' + rm_num + '.h5'))
            '''

        print ("Train Finished.")


    def transform(self, eval_condition="C5", model_use=None, ep_path=None):
        print ("Start i-vectors adaptation.")

        #batch_size = self.batchsize

        # Choose evaluation condition and load enroll("_train") or test dataset folder
        input_folder_enroll = './exp/' + eval_condition + '/ivectors_sre10_train/'
        input_folder_test = './exp/' + eval_condition + '/ivectors_sre10_test/'
        input_folder_test_c5 = './exp/' + eval_condition + '/ivectors_sre10_test_c5/'

        # Load trained Generators model
        #Gs = load_model(model_use)
        Gs = self.combined_Gs
        Gs.load_weights(model_use)

        epoch_folder = str(ep_path)

        for k in range(0, 3):
            if k == 0:
                test_data, test_label = data_prep.datalist_load(foldername=input_folder_test, train=0, use_label=True)
            elif k == 1:
                test_data, test_label = data_prep.datalist_load(foldername=input_folder_test_c5, train=0, use_label=True)
            elif k == 2:
                test_data, test_label = data_prep.datalist_load(foldername=input_folder_enroll, train=0, use_label=True)

            test_data = np.array(test_data)
            test_data = test_data.reshape((-1, 1, model_cyclegan.IVEC_DIM, 1))

            max_data = len(test_data)
            #num_batch = max_data // batch_size

            # Test Loop
            # In application step, set batch_size == 1
            for i in range (0, max_data):
                test_feed = test_data[i]
                test_feed = test_feed.reshape(1, 1, model_cyclegan.IVEC_DIM, 1)

                ### Use G_AB network to obtain adaptive i-vectors_b
                fake_A_from_b = Gs.get_layer('model_4').predict(test_feed)

                fake_A_from_b = fake_A_from_b.reshape((1, model_cyclegan.IVEC_DIM))
                fake_A_from_b_list = fake_A_from_b.tolist()

                if k == 0:
                    output_folder = './exp/' + eval_condition + '/ep' + epoch_folder + '/ivectors_adpt_sre10_test/'
                    data_prep.adpt_ivec2kaldi(fake_A_from_b_list, test_label[i], arkfilepath=output_folder + 'sre10_test.ark')
                elif k == 1:
                    output_folder = './exp/' + eval_condition + '/ep' + epoch_folder + '/ivectors_adpt_sre10_test_c5/'
                    data_prep.adpt_ivec2kaldi(fake_A_from_b_list, test_label[i], arkfilepath=output_folder + 'sre10_test_c5.ark')
                elif k == 2:
                    output_folder = './exp/' + eval_condition + '/ep' + epoch_folder + '/ivectors_adpt_sre10_enroll/'
                    data_prep.adpt_ivec2kaldi(fake_A_from_b_list, test_label[i], arkfilepath=output_folder + 'sre10_enroll.ark')

                print("Saved i-vector {}/{}".format(i+1, max_data))

        print ("Adaptation Finished.")

if __name__ == '__main__':
    gan = CycleGAN()
    to_train = 0
    if to_train > 0:
        epochs = para_setting.set_parameters()['max_epoch']
        gan.train(epochs=epochs)
    else:
        for num_ep in {'50'}:
            modelpath = './models/1902010212/combined_Gs_' + num_ep + '.h5'
            gan.transform(eval_condition="C5", model_use=modelpath, ep_path=num_ep)
