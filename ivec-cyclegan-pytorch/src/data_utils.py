import torch
import kaldi_io  #used on local PC
import numpy as np
import os
import random

def datalist_load(foldername):
    input_data = []
    input_label = []

    scpindex = 'ivector.scp'
    
    for key, mat in kaldi_io.read_vec_flt_scp(os.path.join(foldername, scpindex)):
        matl = mat.tolist()
        input_data.append(matl)
        input_label.append(key)

    return np.array(input_data, dtype=np.float32), input_label

class ReplayBuffer():
    def __init__(self, max_size=256):
        assert (max_size > 0), 'max size should be greater than 0!'
        self.max_size = max_size
        self.data = []

    def push(self,data):
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    self.data[i] = element

    def pop(self, batch_size):
        return_idx= np.random.choice(range(len(self.data)),batch_size,replace=False)
        to_return=[]
        for idx in return_idx:
            to_return.append(self.data[idx])
            
        return torch.cat(to_return)

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