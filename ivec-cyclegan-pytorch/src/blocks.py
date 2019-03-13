import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, nc_input):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.Conv1d(nc_input, nc_input, kernel_size=1, stride=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.InstanceNorm1d(nc_input),
                        nn.Conv1d(nc_input, nc_input, kernel_size=1, stride=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.InstanceNorm1d(nc_input),
                       ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)