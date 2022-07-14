import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
"""
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

#Take from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#Fron another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
#print(f"Ones Tensor: \n {x_ones} \n")

#Random
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
#print(f"Random Tensor: \n {x_rand} \n")
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
prediction = model(data)
loss = (prediction - labels).sum()
loss.backward() # backward pass
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() #gradient descent

prediction1 = model(data)
loss1 = (prediction1 - labels).sum()
loss1.backward() # backward pass
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() #gradient descent

prediction2 = model(data)
loss2 = (prediction2 - labels).sum()
loss2.backward() # backward pass
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() #gradient descent
print('hi')

