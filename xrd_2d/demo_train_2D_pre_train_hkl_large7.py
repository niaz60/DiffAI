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
        self.embedding_layer_h = nn.Embedding(5, 4096)
        self.embedding_layer_k = nn.Embedding(5, 4096)
        self.embedding_layer_l = nn.Embedding(5, 4096)
        self.proj_linear = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
        )
        self.cls = nn.Linear(4096, 230)
        self.cls7 = nn.Linear(4096, 7)
        self.h_predict = nn.Linear(4096, 5)
        self.k_predict = nn.Linear(4096, 5)
        self.l_predict = nn.Linear(4096, 5)
     
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
        x7 = self.cls7(fx)
        # preds_hkl = self.hkl_predict(fx)
        # preds_h, preds_k, preds_l = torch.split(preds_hkl, 1, dim=1)
        preds_h = self.h_predict(fx)
        preds_k = self.k_predict(fx)
        preds_l = self.l_predict(fx)
        return x, x7, preds_h, preds_k, preds_l


"""
Crystal system (7ways)	        space group (230ways)
Cubic	                               195 to 230
Hexagonal	                      168 to 194
Trigonal	                              143 to 167
Tetragonal	                     75 to 142
Orthorhombic	            16 to 74
Monoclinic	                    3 to 15
Triclinic	                            1 to 2 
"""
label_map = {
    0: list(range(194, 230)),
    1: list(range(167, 194)),
    2: list(range(142, 167)),
    3: list(range(74, 142)),
    4: list(range(15, 74)),
    5: list(range(2, 15)),
    6: list(range(0, 2)), 
}


def transform_from_230_to_7(label):
    for key in label_map:
        if label in label_map[key]:
            return key
    return None

# generate a map from 230 to 7
label_map_230_to_7 = {}
for key in label_map:
    for value in label_map[key]:
        label_map_230_to_7[value] = key
        






if __name__ == "__main__":
    root_path = "/scratch/zzh136/xrd_2d/axis20_data/"
    train_path = "/scratch/zzh136/xrd_2d/axis20_data/train_files.xlsx"
    val_path = "/scratch/zzh136/xrd_2d/axis20_data/val_files.xlsx"
    
    label_path = os.path.join(root_path, "label.xlsx")
    train_dataset = Dataset(root_path = root_path+'train/', img_path = train_path, label_path = label_path)
    val_dataset = Dataset(root_path = root_path+'evaluation/', img_path = val_path, label_path = label_path)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=0)
    
    model = models.vgg19(pretrained=False, num_classes=230)
    # backbone_model = models.alexnet(pretrained=True)
    # replace the last layer
    # model = nn.Sequential(*list(backbone_model.children())[:-1], nn.Flatten(), nn.Linear(9216, 230))
    model.load_state_dict(torch.load("/scratch/zzh136/xrd_2d/axis20_data/ckpt/vanilla_vgg19.pth"))
    
    model = HKL_model(model)
    print('wrapped by the HKL model')
    optimazier = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    epochs = 100
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    
    
    model_save_path = os.path.join(root_path, "ckpt", "pretrained_vgg19_hkl_noflip_7.pth")
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        N = 0.
        running_correct7 = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels, h, k, l = data
            label7 = [transform_from_230_to_7(label) for label in labels]
            label7 = torch.tensor(label7)
            label7 = label7.to(device)
            inputs, labels = inputs.to(device), labels.to(device)
            h,k,l=h.to(device), k.to(device), l.to(device)
            N += inputs.size(0)
            optimazier.zero_grad()
            outputs, outputs7 ,ph,pk,pl = model(inputs, h, k, l)
            loss = criterion(outputs, labels) + + 0.1*(criterion(ph, h) + criterion(pk, k) + criterion(pl, l)) + criterion(outputs7, label7)
            loss.backward()
            optimazier.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels.data)
            
            running_correct7 += torch.sum(torch.max(outputs7, 1)[1] == label7.data)
            
            # import pdb; pdb.set_trace()
            tqdm.write(f"Train Epoch {epoch}/{epochs}, batch {i}/{len(train_loader)}, loss: {loss.item()/(i+1)}, acc: {running_correct.double()/N}, acc7: {running_correct7.double()/N}")
        train_loss = running_loss / len(train_loader)
        train_acc = running_correct.double() / N
        running_correct7 = running_correct7.double()
        wandb.log({"train_loss": train_loss, "train_acc": train_acc})
        print(f"Epoch {epoch}/{epochs}, train_loss: {train_loss}, train_acc: {train_acc}, train_acc7: {running_correct7/N}")
        model.eval()
        running_loss = 0.0
        running_correct = 0.0
        running_correct7 = 0.0
        N = 0
        for i, data in enumerate(val_loader, 0):
            inputs, labels,h,k,l = data
            label7 = [transform_from_230_to_7(label) for label in labels]
            label7 = torch.tensor(label7) 
            label7 = label7.to(device) 
            inputs, labels,h,k,l = inputs.to(device), labels.to(device), h.to(device), k.to(device), l.to(device)
            N += inputs.size(0)
            outputs, outputs7, _,_,_ = model(inputs,h,k,l)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels.data)
            running_correct7 += torch.sum(torch.max(outputs7, 1)[1] == label7.data)
            tqdm.write(f"Val Epoch {epoch}/{epochs}, batch {i}/{len(val_loader)}, loss: {loss.item()/(i+1)}, acc: {running_correct.double()/N}, acc7: {running_correct7.double()/N}")
        val_loss = running_loss / len(val_loader)
        val_acc = running_correct.double() / N
        wandb.log({"val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch}/{epochs}, val_loss: {val_loss}, val_acc: {val_acc}, val_acc7: {running_correct7/N}")
    
        torch.save(model.state_dict(), model_save_path)
        print("Model saved to ", model_save_path)
        
    
    
    
