
import torchvision.transforms as transforms
import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable # torch 中 Variable 模块

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[224, 224])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_num', type=int, default=35)
    parser.add_argument('--learning_rate', type=float, default=0.0001)  
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 50])

    x = np.random.rand(3, 200, 176)
    x = torch.from_numpy(x).to(torch.float32)
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)

    model = torchvision.models.vgg19(pretrained=True)
    config = parser.parse_args()
    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1, last_epoch=-1)
    creiteron = torch.nn.CrossEntropyLoss()
    output = model(x)
    label = 1
    label=np.array([1])

    label=torch.from_numpy(label).long()
    loss = creiteron(output, label)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

# 0.00005, 