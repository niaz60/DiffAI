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
from dataloader_hkl import Dataset
from torch.utils.data import  DataLoader
import wandb
from tqdm import tqdm
wandb.init(project="xrd_2d_alexnet")


class HKL_model(nn.Module):
    def __init__(self, model):
        super(HKL_model, self).__init__()
        self.model = model
        # self.feature_extractor = nn.Sequential([self.model.features, self.model.avgpool])
        self.embedding_layer_h = nn.Embedding(3, 4560)
        self.embedding_layer_k = nn.Embedding(3, 4560)
        self.embedding_layer_l = nn.Embedding(3, 4560)
        self.proj_linear = nn.Linear(9216, 4560)
        self.cls = nn.Linear(4560, 230)
        self.h_predict = nn.Linear(4560, 3)
        self.k_predict = nn.Linear(4560, 3)
        self.l_predict = nn.Linear(4560, 3)
     
    def forward(self, x, h, k, l):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x,1)
        h = self.embedding_layer_h(h)
        k = self.embedding_layer_k(k)
        l = self.embedding_layer_l(l)
        position_embedding = h + k + l
        position_featuer = nn.Sigmoid()(position_embedding)
        fx = self.proj_linear(x) * position_featuer
        x = self.cls(fx)
        # preds_hkl = self.hkl_predict(fx)
        # preds_h, preds_k, preds_l = torch.split(preds_hkl, 1, dim=1)
        preds_h = self.h_predict(fx)
        preds_k = self.k_predict(fx)
        preds_l = self.l_predict(fx)
        return x,preds_h, preds_k, preds_l
       



if __name__ == "__main__":
    root_path = "/scratch/zzh136/xrd_2d/redo/redo4/"
    train_path = "/scratch/zzh136/xrd_2d/redo/redo4/train_files.xlsx"
    val_path = "/scratch/zzh136/xrd_2d/redo/redo4/val_files.xlsx"
    
    label_path = os.path.join(root_path, "label.xlsx")
    train_dataset = Dataset(root_path = root_path+'train/', img_path = train_path, label_path = label_path)
    val_dataset = Dataset(root_path = root_path+'evaluation/', img_path = val_path, label_path = label_path)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=0)
    
    model = models.alexnet(pretrained=False, num_classes=230)
    # backbone_model = models.alexnet(pretrained=True)
    # replace the last layer
    # model = nn.Sequential(*list(backbone_model.children())[:-1], nn.Flatten(), nn.Linear(9216, 230))
    model.load_state_dict(torch.load("/scratch/zzh136/xrd_2d/redo/redo4/ckpt/vanilla_alexnet.pth"))
    
    model = HKL_model(model)
    print('wrapped by the HKL model')
    optimazier = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    epochs = 100
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    
    
    model_save_path = os.path.join(root_path, "ckpt", "pretrained_alexnet_hkl_noflip.pth")
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        N = 0.
        for i, data in enumerate(train_loader, 0):
            inputs, labels, h, k, l = data
            inputs, labels = inputs.to(device), labels.to(device)
            h,k,l=h.to(device), k.to(device), l.to(device)
            N += inputs.size(0)
            optimazier.zero_grad()
            outputs,ph,pk,pl = model(inputs, h, k, l)
            loss = criterion(outputs, labels) + + 0.1*(criterion(ph, h) + criterion(pk, k) + criterion(pl, l))
            loss.backward()
            optimazier.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels.data)
            # import pdb; pdb.set_trace()
            tqdm.write(f"Train Epoch {epoch}/{epochs}, batch {i}/{len(train_loader)}, loss: {loss.item()/(i+1)}, acc: {running_correct.double()/N}")
        train_loss = running_loss / len(train_loader)
        train_acc = running_correct.double() / N
        wandb.log({"train_loss": train_loss, "train_acc": train_acc})
        print(f"Epoch {epoch}/{epochs}, train_loss: {train_loss}, train_acc: {train_acc}")
        model.eval()
        running_loss = 0.0
        running_correct = 0.0
        N = 0
        for i, data in enumerate(val_loader, 0):
            inputs, labels,h,k,l = data
            inputs, labels,h,k,l = inputs.to(device), labels.to(device), h.to(device), k.to(device), l.to(device)
            N += inputs.size(0)
            outputs,_,_,_ = model(inputs,h,k,l)
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
        
    
    
    
