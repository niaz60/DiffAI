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
from dataloader_hkl_copy import Dataset
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








if __name__ == "__main__":
    root_path = ["/scratch/zzh136/xrd_2d/axis20_data/ckpt/pretrained_vgg19_hkl_noflip_7.pth","/scratch/zzh136/xrd_2d/model_/model2/model_2_pretrained_vgg19_hkl_noflip_7.pth", "/scratch/zzh136/xrd_2d/model_/model3/model_3_pretrained_vgg19_hkl_noflip_7.pth"]
    
    
    to_merge_model = models.vgg19(pretrained=False, num_classes=230)
    to_merge_model = HKL_model(to_merge_model)
    
    # merge three ckpts
    
    ckpts = [torch.load(path) for path in root_path]
    
    to_merge_model.load_state_dict(ckpts[0])
    
    for key in to_merge_model.state_dict():
        to_merge_model.state_dict()[key] = (ckpts[0][key] + ckpts[1][key] + ckpts[2][key]) / 3
        to_merge_model.state_dict()[key] = to_merge_model.state_dict()[key].cuda()
    
    model_save_path = "/scratch/zzh136/xrd_2d/axis20_data/ckpt/merged_model.pth"
    torch.save(to_merge_model.state_dict(), model_save_path)
        
    
    
    
    
    
    
    # model_save_path = os.path.join(root_path, "ckpt", "model_3_pretrained_vgg19_hkl_noflip_7.pth")
    
        
    
    
    
