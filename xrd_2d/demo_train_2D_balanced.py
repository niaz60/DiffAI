# load AlexNet model and train it on 2D data
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import os
import time
import copy
import numpy as np
from dataloader import Dataset,Balanced_ComplexDataset
from torch.utils.data import  DataLoader
import wandb
from tqdm import tqdm
wandb.init(project="xrd_2d_alexnet")

if __name__ == "__main__":
    root_path = "/scratch/zzh136/xrd_2d/exp5/output"
    train_path = "/scratch/zzh136/xrd_2d/exp5/output/train_files.xlsx"
    val_path = "/scratch/zzh136/xrd_2d/exp5/output/val_files.xlsx"
    
    label_path = os.path.join(root_path, "label_experience5_train_test_eval_10zone.xlsx")
    train_dataset = Balanced_ComplexDataset(root_path = root_path, img_path = train_path, label_path = "/scratch/zzh136/xrd_2d/exp5/output/train_label_map.xlsx", fmt=True)
    val_dataset = Balanced_ComplexDataset(root_path = root_path, img_path = val_path, label_path = "/scratch/zzh136/xrd_2d/exp5/output/val_label_map.xlsx", fmt=True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    
    model = models.alexnet(pretrained=False, num_classes=230)
    optimazier = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    epochs = 100
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model_save_path = os.path.join(root_path, "ckpt", "vanilla_alexnet_balanced.pth")
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        N = 0.
        for i, data in tqdm(enumerate(train_loader, 0)):
            union_names, inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            N += inputs.size(0)
            optimazier.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimazier.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels.data)
            # import pdb; pdb.set_trace()
            tqdm.write(f"train on random Train Epoch {epoch}/{epochs}, batch {i}/{len(train_loader)}, loss: {loss.item()/(i+1)}, acc: {running_correct.double()/N}")
        train_loss = running_loss / len(train_loader)
        train_acc = running_correct.double() / N
        wandb.log({"train_loss": train_loss, "train_acc": train_acc})
        print(f"Epoch {epoch}/{epochs}, train_loss: {train_loss}, train_acc: {train_acc}")
        model.eval()
        running_loss = 0.0
        running_correct = 0.0
        N = 0
        for i, data in tqdm(enumerate(val_loader, 0)):
            union_names, inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            N += inputs.size(0)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels.data)
            tqdm.write(f"Val Epoch {epoch}/{epochs}, batch {i}/{len(val_loader)}, loss: {loss.item()/(i+1)}, acc: {running_correct.double()/N}")
        val_loss = running_loss / len(val_loader)
        val_acc = running_correct.double() / N
        wandb.log({"val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch}/{epochs}, val_loss: {val_loss}, val_acc: {val_acc}")
    
        torch.save(model.state_dict(), model_save_path)
        print("Model saved to ", model_save_path)
        
    
    
    