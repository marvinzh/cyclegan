import torch
from torch import nn


class CycleGAN(nn.Module):
    def __init__(self, g_s2t, g_t2s, d_src, d_trg):
        super().__init__()
        self.g_s2t = g_s2t
        self.g_t2s = g_t2s
        self.d_src = d_src
        self.d_trg = d_trg

    def generator_step(self):
        pass
    
    def discriminator_step(self):
        pass
    
    
