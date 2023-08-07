import math
import os
import random
import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

from model import *



if __name__ == "__main__":
    datapath = './data'
    train_dataset = XRD(datapath, train=True)
    val_dataset = XRD(datapath, train=False)
    
    
    task = 'cnn'
    num_class = 7
    batch_size = 32
    learning_rate = 1e-3
    epochs = 100
    if task == 'cnn':
        model_cnn = CNN()
    elif task == 'nopool':
        model_cnn = NoPoolCNN()
    model = nn.Sequential(model_cnn,Predictor(num_class))
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device('cuda')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    with tqdm.tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Epoch {epoch+1}/{epochs} (Train) | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
            for i, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                pbar.set_description(f"Epoch {epoch+1}/{epochs} (Val) | Batch {i+1}/{len(val_loader)} | Loss: {loss.item():.4f}")
            pbar.update(1)
    
    model_save_path = '{}_{}'.format(task, num_class)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(model.state_dict(), '{}/model.pth'.format(model_save_path))
    