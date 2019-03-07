import torch
import kaldi_io  #used on local PC
import numpy as np
import os

def datalist_load(foldername):
    input_data = []
    input_label = []

    scpindex = 'ivector.scp'
    
    for key, mat in kaldi_io.read_vec_flt_scp(foldername + scpindex):
        matl = mat.tolist()
        input_data.append(matl)
        input_label.append(key)

    return np.array(input_data), input_label
