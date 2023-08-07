import math
import os
import random

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset



class NoPoolCNN(nn.Module):
    def __init__(self, input_shape=(1,)):
        super().__init__()

        in_channels = input_shape if isinstance(input_shape, int) else input_shape[0]

        self.CNN = \
            nn.Sequential(
                nn.Conv1d(in_channels, 80, 100, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(80, 80, 50, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(80, 80, 25, 2),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

    def forward(self, obs):
        return self.CNN(obs)


class CNN(nn.Module):
    def __init__(self, input_shape=(1,)):
        super().__init__()

        in_channels = input_shape if isinstance(input_shape, int) else input_shape[0]

        self.CNN = \
            nn.Sequential(
                nn.Conv1d(in_channels, 80, 100, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.AvgPool1d(3, 2),
                nn.Conv1d(80, 80, 50, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.AvgPool1d(3),
                nn.Conv1d(80, 80, 25, 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.AvgPool1d(3),
            )

    def forward(self, obs):
        return self.CNN(obs)


class Predictor(nn.Module):
    def __init__(self, input_shape=(1024,), output_shape=(7,)):
        super().__init__()

        input_dim = input_shape if isinstance(input_shape, int) else math.prod(input_shape)
        output_dim = output_shape if isinstance(output_shape, int) else math.prod(output_shape)

        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(input_dim, 2300), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(2300, 1150), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(1150, output_dim))

    def forward(self, obs):
        return self.MLP(obs)


class MLP(nn.Module):
    def __init__(self, input_shape=(8500,), output_shape=(7,)):
        super().__init__()

        in_channels = input_shape if isinstance(input_shape, int) else math.prod(input_shape)
        output_dim = output_shape if isinstance(output_shape, int) else math.prod(output_shape)

        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(in_channels, 4000), nn.ReLU(), nn.Dropout(0.6),
                                 nn.Linear(4000, 3000), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(3000, 1000), nn.ReLU(), nn.Dropout(0.4),
                                 nn.Linear(1000, 800), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(800, output_dim))

    def forward(self, obs):
        return self.MLP(obs)


class XRD(Dataset):
    def __init__(self, roots, train=True, num_classes=7, seed=0, train_eval_splits=None):



        self.indices = []
        self.features = {}
        self.labels = {}

        features_path = roots + "features.csv"
        label_path = roots + f"labels{num_classes}.csv"
        self.classes = list(range(num_classes))
        print(f'Loading [root={roots}, split={train_eval_splits if train else 1 - train_eval_splits}, train={train}] to CPU...')
        # Store on CPU
        with open(features_path, "r") as f:
            self.features = f.readlines()
        with open(label_path, "r") as f:
            self.labels = f.readlines()
            full_size = len(self.labels)
        print('Data loaded ✓')
        train_size = round(full_size * train_eval_splits)
        full = range(full_size)
        # Each worker shares an indexing scheme
        random.seed(seed)
        train_indices = random.sample(full, train_size)
        eval_indices = set(full).difference(train_indices)
        indices = train_indices if train else eval_indices
        


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        x = torch.FloatTensor(list(map(float, self.features[idx].strip().split(','))))
        y = np.array(list(map(float, self.labels[idx].strip().split(',')))).argmax()

        return x, y

