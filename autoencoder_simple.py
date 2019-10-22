from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.fce1 = nn.Linear(384, 300)
        self.fce2 = nn.Linear(300, 200)
        self.fce3 = nn.Linear(200, 128)
        self.fcd2 = nn.Linear(128, 200)
        self.fcd3 = nn.Linear(200, 300)
        self.fcd4 = nn.Linear(300, 384)

    def encode(self, x):
        x = F.leaky_relu(self.fce1(x))
        x = F.leaky_relu(self.fce2(x))
        x = F.leaky_relu(self.fce3(x))
        return x

    def decode(self, z):
        z = F.leaky_relu(self.fcd2(z))
        z = F.leaky_relu(self.fcd3(z))
        z = F.sigmoid(self.fcd4(z))
        return z

    def forward(self, x):
        enc = self.encode(x.view(-1, 384))
        return self.decode(enc), enc


def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x)
    return BCE

