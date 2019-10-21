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
        self.bne1 = nn.BatchNorm1d(300)
        self.fce2 = nn.Linear(300, 200)
        self.bne2 = nn.BatchNorm1d(200)
        self.fce3 = nn.Linear(200, 100)
        self.bne3 = nn.BatchNorm1d(100)
        self.fce4 = nn.Linear(100, 50)
        self.bne4 = nn.BatchNorm1d(50)

        self.fcd1 = nn.Linear(50, 100)
        self.bnd1 = nn.BatchNorm1d(100)
        self.fcd2 = nn.Linear(100, 200)
        self.bnd2 = nn.BatchNorm1d(200)
        self.fcd3 = nn.Linear(200, 300)
        self.bnd3 = nn.BatchNorm1d(300)
        self.fcd4 = nn.Linear(300, 384)
        self.bnd4 = nn.BatchNorm1d(384)

    def encode(self, x):
        x = F.leaky_relu(self.bne1(self.fce1(x)))
        x = F.leaky_relu(self.bne2(self.fce2(x)))
        x = F.leaky_relu(self.bne3(self.fce3(x)))
        x = F.leaky_relu(self.bne4(self.fce4(x)))
        return x

    def decode(self, z):
        z = F.leaky_relu(self.bnd1(self.fcd1(z)))
        z = F.leaky_relu(self.bnd2(self.fcd2(z)))
        z = F.leaky_relu(self.bnd3(self.fcd3(z)))
        z = F.sigmoid(self.bnd4(self.fcd4(z)))
        return z

    def forward(self, x):
        enc = self.encode(x.view(-1, 384))
        return self.decode(enc), enc


def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 384), size_average=False)
    return BCE

