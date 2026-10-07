import torch
import numpy as np


"""
data = [[1, 2],[3, 4]]
#x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x = torch.zeros(3, 4)  
xm = torch.tensor([1, 2, 3]) 

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

print(x_np)
print(x)
print(xm)



# Datasets & DataLoaders

# 1. Loading a Datase

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

"""