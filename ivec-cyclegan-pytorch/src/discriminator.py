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

        self.model = nn.Sequential(*model)
        self.fc1=nn.Linear(12800,512)
        self.fc2=nn.Linear(512,512)
        self.out = nn.Linear(512,1)
        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, x):
        x =  self.model(x)
        x = x.view(x.shape[0], -1)
        a1 = self.leaky_relu(self.fc1(x))
        a2 = self.leaky_relu(self.fc2(a1))
        return self.out(a2)
        
        # Average pooling and flatten
        # return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)