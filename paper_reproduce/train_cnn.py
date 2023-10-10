import math
import os
import random

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
import math
import os
import random
import tqdm
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
    def __init__(self, input_shape=(12160,), output_shape=(7,)):
        super().__init__()

        input_dim = input_shape if isinstance(input_shape, int) else math.prod(input_shape)
        output_dim = output_shape if isinstance(output_shape, int) else math.prod(output_shape)

        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(input_dim, 2300), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(2300, 1150), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(1150, output_dim))

    def forward(self, obs):
        # print(obs.shape)
        # import pdb; pdb.set_trace()
        return self.MLP(obs)




class XRD(Dataset):
    def __init__(self, roots, train=True, num_classes=7, seed=0, train_eval_splits=0.7):



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
        rruff_root =  '/scratch/zzh136/data/xrd/rruff/'
        rruff_features_path = rruff_root + "features.csv"
        rruff_label_path = rruff_root + f"labels{num_classes}.csv"
        with open(rruff_features_path, "r") as f:
            self.features += f.readlines()
        with open(rruff_label_path, "r") as f:
            self.labels += f.readlines()
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
    
    
if __name__ == "__main__":
    datapath = '/scratch/public/jsalgad2/icsd1.2m_large/' # '/scratch/zzh136/data/xrd/database_xrds/HighRes2Theta_5to90/ExampleSet/'# '/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/'
    train_dataset = XRD(datapath, train=True)
    val_dataset = XRD(datapath, train=False)
    
    
    task = 'large_cnn'
    num_class = 7
    batch_size = 1024
    learning_rate = 1e-3
    epochs = 100
    if 'cnn' in task:
        model_cnn = CNN()
    elif 'nopool' in task:
        model_cnn = NoPoolCNN()
    model = nn.Sequential(model_cnn,Predictor(input_shape=400, output_shape=num_class))
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device('cuda')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    # import pdb; pdb.set_trace()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    with tqdm.tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            train_n = 0.
            train_correct = 0.
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device).unsqueeze(1)
                y = y.to(device)
                train_n += y.size(0)
                # import pdb; pdb.set_trace()
                optimizer.zero_grad()
                y_pred = model(x)
                train_correct += (y_pred.argmax(dim=1) == y).float().sum()
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Epoch {epoch+1}/{epochs} (Train) | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {train_correct/train_n:.4f}")
            val_n = 0.
            val_correct = 0.
            for i, (x, y) in enumerate(val_loader):
                x = x.to(device).unsqueeze(1)
                y = y.to(device)
                val_n += y.size(0)
                y_pred = model(x)
                val_correct += (y_pred.argmax(dim=1) == y).float().sum()
                loss = criterion(y_pred, y)
                pbar.set_description(f"Epoch {epoch+1}/{epochs} (Val) | Batch {i+1}/{len(val_loader)} | Loss: {loss.item():.4f} | Acc: {val_correct/val_n:.4f}")
            pbar.update(1)
    
            model_save_path = '{}_{}'.format(task, num_class)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), '{}/model.pth'.format(model_save_path))
