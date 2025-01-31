import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
from pytorch_metric_learning import losses
import torch.optim as optim
import sys
sys.path.insert(1, 'D:/Python script/transformer/lsfb_transfo/transforms')
sys.path.insert(1, 'D:/Python script/transformer/lsfb_transfo/loader')
sys.path.insert(1, 'D:/Python script/transformer/lsfb_transfo/models')
import encoder, simclr
import augmentation 
import load_data
from lightly.loss import NTXentLoss
import lsfb_dataset
import matplotlib.pyplot as plt
from lsfb_dataset import LSFBIsolConfig, LSFBIsolLandmarks
backbone = encoder.ViTModel(1,10,32,1024,4,1024,10,0.1).to('cuda')
simclr_model = simclr.SimCLR(backbone).to('cuda')
dataset = LSFBIsolLandmarks(LSFBIsolConfig(
    root="C:/Users/abassoma/Documents/LSFB_Dataset/lsfb_dataset/isol",
    landmarks=('pose','left_hand', 'right_hand'),
    split = "mini_sample",
    sequence_max_length=50,
    show_progress=True,
))

print(len(dataset))

from torch.utils.data import random_split

# Taille du dataset
dataset_size = len(dataset)

# Calcul des longueurs des splits
split_2_3 = int(dataset_size / 2)
split_1_3 = dataset_size - split_2_3

# Division du dataset
dataset_2_3, dataset_1_3 = random_split(dataset, [split_2_3, split_1_3])
unsupervised_trainloader = load_data.CustomDataset.build_dataset(dataset_2_3)
supervised_trainloader = load_data.CustomDataset.build_dataset(dataset_1_3)
# train_loader = load_data.CustomDataset.build_dataset(dataset)
# x = next(iter(train_loader))[0]

criterion = NTXentLoss()
optimizer = optim.SGD(simclr_model.parameters(), lr=0.001)

def contrastive_training(train_loader, epochs):
    epoch_losses = [] 
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (data, target) in enumerate(tqdm(train_loader)):
            data = data.to('cuda')
            z1, z2 =  simclr_model.forward(data)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.legend()
    plt.savefig('Training_loss_simclr_others_essai_4.png')
    plt.show()
    
contrastive_training(unsupervised_trainloader, 10)
