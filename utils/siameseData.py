

import torch
import numpy as np
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.nn.modules.loss import TripletMarginLoss
from itertools import permutations, product
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# from pytorch_metric_learning.utils.accuracy_calculator import  AccuracyCalculator

mean, std = 0.1307, 0.3081 #MNIST
preprocess =  transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)) ])



class CustomSiamesesDataset(Dataset):
    def __init__(self, X, y,transform=None, augment=False):
        self.text = X
        self.labels = y
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        texts = torch.tensor(self.text[idx])
        labels = torch.tensor(self.labels[idx])
        sample = (texts, labels)
        return sample

class SiameseDataset(Dataset):
    def __init__(self, dataset, batch_size=2, transform=None):
        self.A = torch.tensor([])
        self.B = torch.tensor([])

        self.Labels = []
        self.batch_size = batch_size
        self.transform = transform

        samples, lab_set, min_size = self.split_by_label(dataset)

        self.batch_size = min(self.batch_size, min_size)

        lab_set_prod = list(product(lab_set, repeat=2))
        np.random.shuffle(lab_set_prod)

        for i, j in lab_set_prod:
            a, b = self.Pairs_maker(samples[i], samples[j], i == j)
            self.A = torch.cat((self.A, a), 0)
            self.B = torch.cat((self.B, b), 0)

            self.Labels += [[0] if i == j else [1] for _ in range(self.batch_size)]

        print(f"Number of labels permutations: {len(lab_set_prod)}")
        print(f"Pair samples per permutations: {self.batch_size}")
        print(f"Total number of pair samples: {len(self.A)}")

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        A_sample = self.A[idx]
        B_sample = self.B[idx]
        landmarks = torch.tensor(self.Labels)[idx]

        if self.transform:
            A_sample = self.transform(A_sample)
            B_sample = self.transform(B_sample)
            landmarks = self.transform(landmarks)

        return (A_sample, B_sample), landmarks

    def split_by_label(self, dataset):
        labels_set = list(range(10))

        samples_by_label = {}
        label_size = []
        for label in labels_set:
            samples_by_label[label] = dataset[dataset.labels== label]

            l, w, d = samples_by_label[label].shape
            label_size.append(l)

            samples_by_label[label] = samples_by_label[label].view(l, 1, d, w)

        return samples_by_label, labels_set, np.min(label_size)//2

    def Pairs_maker(self, class_1, class_2, same_class):
        if same_class:
            index_a = np.random.choice(range(len(class_1)), self.batch_size * 2, replace=False)

            a = class_1[index_a[:self.batch_size]]
            b = class_1[index_a[self.batch_size:]]
        else:
            index_a = np.random.choice(range(len(class_1)), self.batch_size, replace=False)
            index_b = np.random.choice(range(len(class_2)), self.batch_size, replace=False)

            a = class_1[index_a]
            b = class_2[index_b]

        return a, b

# #Load Data
# train_dataset = MNIST(root='dataset/', train=True, transform=preprocess, download='True')
# test_dataset = MNIST(root='dataset/', train=False, transform=preprocess, download='True')

# #Data to pairs format
# batch_size = 512
# siamese_train_ds = SiameseDataset(dataset=train_dataset, batch_size=batch_size)

# # Create validation datasets
# val_size = int(batch_size * 0.3)
# print("\nValidation batch size: ", val_size)
# siamese_val_ds = SiameseDataset(dataset=train_dataset, batch_size=val_size)

# #Dataset to Batches
# siamese_train_ld = DataLoader(siamese_train_ds, batch_size=batch_size, shuffle=True)
# siamese_val_ld = DataLoader(siamese_val_ds, batch_size=val_size, shuffle=False)
