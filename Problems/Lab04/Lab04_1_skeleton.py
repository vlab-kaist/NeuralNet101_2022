import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

def main():
    num_epochs = 5
    batch_size = 100

    mnist_train = dsets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

    #Write your code here
    #Do not use mnist_test to train model

    for epoch in range(num_epochs):
        for X, Y in data_loader:
            #Write your code here

    return model
