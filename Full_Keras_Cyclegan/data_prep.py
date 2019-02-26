#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#from . import kaldi_io #used on pikaia
import kaldi_io  #used on local PC
import numpy as np
import os
import random
import tensorflow as tf

'''
# This function is only used while using 40 splices ark/scp files of ivectors.
def create_list(foldername, fulldir=True, feattype='ivector.', suffix='.ark'):
    file_list_tmp = os.listdir(foldername)
    file_list = []
    if fulldir:
        for item in file_list_tmp:
            if feattype in item:
                if suffix in item:
                    file_list.append(os.path.join(foldername, item))
    else:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(item)
    
    return file_list
'''

def datalist_load(foldername, suffix='.scp', train=1, use_label=False):
    input_data = []
    input_label = []

    scpindex = 'ivector.scp'
    
    for key,mat in kaldi_io.read_vec_flt_scp(foldername + scpindex):
        matl = mat.tolist()
        #print (matl)
        input_data.append(matl)
        input_label.append(key)

    # Shuffle the training data.
    if train == 1:
        random.shuffle(input_data)
        return input_data
    
    # Do not shuffle the evaluation data.
    elif train == 0:
        if use_label:
            return input_data, input_label
        else:
            return input_data

def data_augment(foldername, picknum=1000, aug_times=10):
    input_data = []

    scpindex = 'ivector.scp'
    
    for key,mat in kaldi_io.read_vec_flt_scp(foldername + scpindex):
        input_data.append(mat)
    
    #pick out 1000 from MIXER as the adapt subset.
    indata = input_data[0: picknum]
    temp = indata

    #do augmentation
    for i in range(0, aug_times-1):
        random.shuffle(temp)
        indata = indata + temp
    
    return indata

def adpt_ivec2kaldi(data, label, arkfilepath='./default_ivec.ark'):
    # This function writes the output i-vectors from CycleGAN's generator into ark files.
    # the format of created files corresponds to ivector's ark files in Kaldi.

    #output_ivec_a = []
    #This part is for single-line ivector.
    file = open(arkfilepath, 'a+')
    temp_label = str(label)
    '''
    data9 = []
    data_1dim = data[0]
    for i in range (0, len(data_1dim)):
        temp = round(data_1dim[i], 9)
        data9.append(temp)
    '''
    temp_data1 = str(data).strip("[]")
    temp_data2 = temp_data1.replace(',', '')
    temp_ivec = temp_label + '  [ ' + temp_data2 + ' ]'
    #output_ivec_a.append(temp_ivec)
    file.write(temp_ivec + '\n')
    file.close()

    '''
    #This part is for multi-line ivectors.
    for i in range (0, len(data)):
        temp_label = str(label[i]).strip("'")
        temp_data1 = str(data[i]).strip("[]")
        temp_data2 = temp_data1.replace(',', '')
        temp_ivec = temp_label + '  [ ' + temp_data2 + ' ]'
        output_ivec_a.append(temp_ivec)
    
    file = open(arkfilepath, 'a+')
    for j in range(0, len(output_ivec_a)):
        file.write(str(output_ivec_a[j]) + '\n')
    file.close()
    '''
