from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class ThaiLife(nn.Module):
    def __init__(self):
        super(ThaiLife, self).__init__()

        self.fce1 = nn.Linear(384, 300)
        self.fce2 = nn.Linear(300, 200)
        self.fce3 = nn.Linear(200, 128)

        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)

    def encode(self, x):
        x = F.leaky_relu(self.fce1(x))
        x = F.leaky_relu(self.fce2(x))
        x = F.leaky_relu(self.fce3(x))
        return x
       
    def forward(self, x):
        enc = self.encode(x.view(-1, 384))
        x = F.leaky_relu(self.fc1(enc))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


def loss_function(recon_x, x):
    CE = F.cross_entropy(recon_x, x.view(-1, 3), size_average=False)
    return CE

