import torch.nn as nn
import torch.nn.functional as F
from blocks import ResidualBlock
import torch


class Generator(nn.Module):
    def __init__(self, conv_channels,trans_channels, kernels, strides, n_residual_blocks=9, leaky_slope=0.2):
        super(Generator, self).__init__()

        models=[]
        for i in range(len(conv_channels)-1):
            models+=[
                nn.Conv1d(conv_channels[i],conv_channels[i+1],kernel_size=kernels[i], stride=strides[i], padding=1),
                nn.InstanceNorm1d(conv_channels[i+1]),
                nn.ReLU(inplace=True)
            ]
            
        
        for _ in range(n_residual_blocks):
            models+=[
                ResidualBlock(conv_channels[-1],leaky_slope=leaky_slope)
            ]
        
        kernels=kernels[len(conv_channels):]
        strides=strides[len(conv_channels):]
        for i in range(len(trans_channels)-1):
            models+=[
                nn.ConvTranspose1d(trans_channels[i],trans_channels[i+1],kernel_size=kernels[i], stride=strides[i], padding=1, output_padding=1),
                nn.InstanceNorm1d(trans_channels[i+1]),
                nn.ReLU(inplace=True)
            ]

        models+=[
            nn.Conv1d(trans_channels[-1],conv_channels[0], kernel_size=kernels[-1], stride=strides[-1], padding=1),
            nn.InstanceNorm1d(conv_channels[0]),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*models)


        # # downsampling
        # self.c1 = nn.Conv1d(channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        # self.c2 = nn.Conv1d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)
        # self.c3 = nn.Conv1d(channels[2], channels[3], kernel_size=3, stride=2, padding=1)


        # # res blocks
        # self.res_blocks= nn.ModuleList()
        # for _ in range(n_residual_blocks):
        #     self.res_blocks.append(ResidualBlock(128,))

        # # upsampling
        # self.tc1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.tc2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.c4 = nn.Conv1d(32,nc_output, kernel_size=3,stride=1,padding=1)
        
        # self.leaky_relu=nn.LeakyReLU(negative_slope=0.2)

        # # Initial convolution block       
        # model = [
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(nc_input, 64, 7),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True)
        #     ]

        # # Downsampling
        # in_features = 64
        # out_features = in_features * 2
        # for _ in range(2):
        #     model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
        #                 nn.InstanceNorm2d(out_features),
        #                 nn.ReLU(inplace=True) ]
        #     in_features = out_features
        #     out_features = in_features*2

        # # Residual blocks
        # for _ in range(n_residual_blocks):
        #     model += [ResidualBlock(in_features)]

        # # Upsampling
        # out_features = in_features//2
        # for _ in range(2):
        #     model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
        #                 nn.InstanceNorm2d(out_features),
        #                 nn.ReLU(inplace=True) ]
        #     in_features = out_features
        #     out_features = in_features//2

        # # Output layer
        # model += [  nn.ReflectionPad2d(3),
        #             nn.Conv2d(64, nc_output, 7),
        #             nn.Tanh() ]

        # self.model = nn.Sequential(*model)

    def forward(self, x):

        # x: (B, 1,600)
        # a1 = self.leaky_relu(self.c1(x))
        # # a1: (B, 32, 600)
        # a2 = self.leaky_relu(self.c2(a1))
        # # a2: (B, 64, 300)
        # a3 = self.leaky_relu(self.c3(a2))
        # # a3: (B, 128, 150)

        # # print(a3.shape)
        # a4 = a3
        # for block in self.res_blocks:
        #     a4= block(a4)
        
        # # print(a4.shape)
        # # a4: (B, 128, 150)
        # a5 = self.leaky_relu(self.tc1(a4))
        # # a5: (B, 64, 300)
        # a6 = self.leaky_relu(self.tc2(a5))
        # # a6 :(B, 32, 600)
        # a7 = self.leaky_relu(self.c4(a6))
        # # a7: (B, 1, 600)
        # return a7


        
        return self.model(x)