import torch
import kaldi_io  #used on local PC
import numpy as np
import os
import random

def datalist_load(foldername):
    input_data = []
    input_label = []

    scpindex = 'ivector.scp'
    
    for key, mat in kaldi_io.read_vec_flt_scp(foldername + scpindex):
        matl = mat.tolist()
        input_data.append(matl)
        input_label.append(key)

    return np.array(input_data), input_label

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)