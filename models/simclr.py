import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
sys.path.insert(1, 'D:/Python script/transformer/lsfb_transfo/transforms')
sys.path.insert(1, 'D:/Python script/transformer/lsfb_transfo/loader')
sys.path.insert(1, 'D:/Python script/transformer/lsfb_transfo/models')
import encoder
import augmentation 
import load_data
import utils

class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.base_encoder = base_encoder
        self.projection = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, projection_dim)
        )
        
    @staticmethod
    def generate_pairs(x):
        x1 =  augmentation.apply_random_augmentation(x)
        x2 = augmentation.apply_random_augmentation(x)
        return x1, x2
        
    def forward(self, x):
        x1, x2 = self.generate_pairs(x)
        x1 = x1.to('cuda')
        x2 = x2.to('cuda')
        h1 = self.base_encoder(x1)
        h2 = self.base_encoder(x2)
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        return z1, z2

