import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import random
import json

class mydataset(Dataset):
    def __init__(self, transform = None, used_data='train/'):
        self.train_dir = os.sep.join(['dataset', used_data])
        self.transform = transform
        self.samples, self.annotions = self._load_samples()
        speed = np.zeros([200, 176])
        with open('pos2id.json', 'r') as f:
            pos2id = json.load(f)
            f.close()
        for i in range(speed.shape[0]):
            for j in range(speed.shape[1]):
                if i == 199 and j < 11:
                    continue
                speed[i][j] = pos2id[str(i)+','+str(j)][1]
        self.speed = speed
    
    def _load_samples(self):
        samples, annotions = [], []
        for f in os.listdir(self.train_dir):
            with open(os.sep.join([self.train_dir, f]),'r') as fi:
                load_dict = json.load(fi)
            # print(load_dict)
            for k in range(len(load_dict)):
                tmp_s = [load_dict[k]['cond1'], load_dict[k]['cond2']]
                tmp_a = load_dict[k]['label']
                samples.append(tmp_s)
                annotions.append(tmp_a)
        return samples, annotions

    def __getitem__(self, index):
        data, label = self.samples[index], self.annotions[index]
        with open('id2pos.json','r') as f:
            id2pos = json.load(f)
            f.close()
        with open('pos2id.json', 'r') as f:
            pos2id = json.load(f)
            f.close()
        img = np.zeros([3, 200, 176])
        for k in data[0]:
            [i, j] = id2pos[str(k)]
            img[0][i][j] = 1
        for k in data[1]:
            [i, j] = id2pos[str(k)]
            img[1][i][j] = 1
        img[2] = self.speed
        return img, label
    
    def __len__(self):
        return len(self.samples)