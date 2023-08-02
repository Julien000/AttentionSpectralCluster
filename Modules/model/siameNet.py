import torch
from torch import nn

class SiameseNet(nn.Module):
    def __init__(self, **kwargs):
        super(SiameseNet, self).__init__()
        self.s = nn.Sequential( 
            nn.Linear(kwargs["input_size"],1024), 
             nn.ReLU(),
             nn.Linear(1024, 1024), 
             nn.ReLU(),
             nn.Linear(1024, 512), 
             nn.ReLU(),
             nn.Linear(512,kwargs["output_size"]),
             nn.ReLU())

    def forward(self, x1, x2=None):
        if x2 == None:
            z1 = self.s(x1)
            return z1
        else:
            z1 = self.s(x1)
            z2 = self.s(x2)
            return z1, z2