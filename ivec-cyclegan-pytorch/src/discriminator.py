import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [
            nn.Conv1d(input_nc, 64, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
            ]

        model += [
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True)
            ]
        # (B,128,100)
        

        # FCN classification layer
        model += [
            nn.Conv1d(128, 512, kernel_size=100,padding=1)
            ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)