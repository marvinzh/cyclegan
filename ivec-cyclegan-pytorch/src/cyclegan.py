import torch.nn as nn
import torch.nn.functional as F
import torch

import argparse
import itertools
from torch.utils.data import DataLoader

from generator import Generator
from discriminator import Discriminator

import hparams as C

if __name__ == "__main__":

    g_s2t = Generator(C.nc_input, C.nc_output, C.n_res_block)
    g_t2s = Generator(C.nc_input, C.nc_output, C.n_res_block)
    d_s = Discriminator(C.input_nc)
    d_t = Discriminator(C.input_nc)

    if C.use_cuda:
        g_s2t.cuda()
        g_t2s.cuda()
        d_s.cuda()
        d_t.cuda()
    