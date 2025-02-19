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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# wandb.init(project="xrd_2d_alexnet")


import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_path', type=str, default='alexnet', help='model name')
argparser.add_argument('--test_path', type=str, default='alexnet', help='model name')

args = argparser.parse_args()


def accuracy_topk(output, target, maxk):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    # import pdb;pdb.set_trace()
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # import pdb;pdb.set_trace()
    correct = pred.eq(target.cuda().view(1, -1).expand_as(pred))

    res = correct[:maxk].reshape(-1).float().sum(0)
    return res


class HKL_model(nn.Module):
    def __init__(self, model):
        super(HKL_model, self).__init__()
        self.model = model
        self.embedding_layer_h = nn.Embedding(5, 4096)
        self.embedding_layer_k = nn.Embedding(5, 4096)
        self.embedding_layer_l = nn.Embedding(5, 4096)
        self.proj_linear = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),                                                                                                          nn.Dropout(p=0.5))
        self.cls = nn.Linear(4096, 230)
        self.h_predict = nn.Linear(4096, 5)
        self.k_predict = nn.Linear(4096, 5)
        self.l_predict = nn.Linear(4096, 5)
        self.cls7 = nn.Linear(4096, 7)
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
        ph = self.h_predict(fx)
        pk = self.k_predict(fx)
        pl = self.l_predict(fx)
        return x, x7, ph, pk, pl



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
    root_path = args.test_path
    val_path = args.test_path + "/test_files.xlsx"
    
    print('begin initialization')
    label_path = os.path.join(root_path, "label.xlsx")
    # train_dataset = Dataset(root_path = root_path, img_path = train_path, label_path = label_path)
    val_dataset = Dataset(root_path = root_path+'/data/', img_path = val_path, label_path = label_path, fmt=True)
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    print('build val dataset ok')
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=0)
    print('build val loader ok')
    
    model = models.vgg19(pretrained=False, num_classes=230)
    optimazier = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    model=HKL_model(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    epochs = 100
    batch_size = 256
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    
    
    # statistic the accuracy for each class
    class_correct = list(0. for i in range(230))
    class_total = list(0. for i in range(230))
    class_correct = np.array(class_correct)
    class_total = np.array(class_total)
    class_correct = torch.tensor(class_correct)
    class_total = torch.tensor(class_total)
    class_correct = class_correct.to(device)
    class_total = class_total.to(device)
    class_entropy = list(0. for i in range(230))
    class_entropy = np.array(class_entropy)
    class_entropy = torch.tensor(class_entropy)
    model.eval()
    running_loss = 0.0
    running_correct = 0.0
    
    confusion_matrix = torch.zeros(230, 230)
    confusion_matrix7 = torch.zeros(7, 7)
    
    class_total7 = torch.zeros(7)
    class_correct7 = torch.zeros(7)
    
    
    entropy_list_w_c = [[],[]]
    N = 0
    running_correct_5 = 0.0
    confusion_matrix = torch.zeros(230, 230)
    correct_h = 0
    correct_k = 0
    correct_l = 0
    all_correct = 0
    running_correct7 = 0.0
    running_correct7_5 = 0.0
    for i, data in tqdm(enumerate(val_loader, 0)):
        inputs, labels, h,k,l, fn = data
        
        label7 = [transform_from_230_to_7(label) for label in labels]
        label7 = torch.tensor(label7)
        label7 = label7.to(device)
        
        
        
        inputs, labels = inputs.to(device), labels.to(device)
        h,k,l = h.to(device), k.to(device), l.to(device)
        N += inputs.size(0)
        outputs,outputs7, ph_logit,pk_logit,pl_logit = model(inputs,h,k,l)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        labels = labels.cpu()
        preds = preds.cpu()
        entropies = -torch.sum(outputs * torch.log(outputs), dim=1).detach().cpu().numpy()
        # import pdb;pdb.set_trace()
        top_5acc = accuracy_topk(outputs, labels, 3)
        running_correct_5 += top_5acc
        running_correct += torch.sum(preds == labels.data)
        
        # verfiy whehter the ph, pk, pl is correct
        ph = torch.argmax(ph_logit, dim=1)
        pk = torch.argmax(pk_logit, dim=1)
        pl = torch.argmax(pl_logit, dim=1)
        correct_h += torch.sum(ph == h)
        correct_k += torch.sum(pk == k)
        correct_l += torch.sum(pl == l)
        # import pdb;pdb.set_trace()
        all_correct += torch.sum((ph == h) & (pk == k) & (pl == l))
        running_correct7 += torch.sum(torch.max(outputs7, 1)[1] == label7.data)
        running_correct7_5+= accuracy_topk(outputs7,label7.data,3)
        for t, p, e in zip(labels.view(-1), preds.view(-1), entropies):
            confusion_matrix[t.long(), p.long()] += 1
            class_total[t.long()] += 1
            class_correct[t.long()] += p.cpu() == t.cpu()
            # entropy_list_w_c[p==t].append(e)
        
        for t, p in zip(label7.view(-1), torch.max(outputs7, 1)[1].view(-1)):
            confusion_matrix7[t.long(), p.long()] += 1
            class_total7[t.long()] += 1
            class_correct7[t.long()] += p.cpu() == t.cpu()
        
        
        # tqdm.write(f"Val Epoch {epoch}/{epochs}, batch {i}/{len(val_loader)}, loss: {loss.item()/(i+1)}, acc: {running_correct.double()/N}")
    val_loss = running_loss / len(val_loader)
    val_acc = running_correct.double() / N
    val_acc5 = running_correct_5/N
    print('accuracy: {}'.format(val_acc))
    print('accuracy3: {}'.format(val_acc5))
    print("accuracy for 7-way classification: {}".format(running_correct7.double()/N))
    print('accuracy3: {}'.format(running_correct7_5.double()/N))
    class_ratio = class_total / N
    wac = 0.
    nc = 0.
    
    
    for c in range(230):
        if class_total[c]!=0:
            wac +=  class_correct[c] / class_total[c]
            nc+=1
    print('average accuracy: {}'.format(wac/nc))
    
    with open("re_exp/result/accuracy_{}_{}.txt".format(args.model_path.split('/')[-1], args.test_path.split('/')[-1]), "w") as f:
        f.write("accuracy: {}\n".format(val_acc))
        f.write("accuracy3: {}\n".format(val_acc5))
        f.write("average accuracy: {}\n".format(wac/nc))
        f.write("accuracy for 7-way classification: {}\n".format(running_correct7.double()/N))
        f.write("top 3 accuracy for 7-way classification: {}\n".format(running_correct7_5.double()/N))
    
    # save the per class accuracy to excel
    
    import pandas as pd
    class_total = class_total.cpu().numpy()
    class_correct = class_correct.cpu().numpy()
    class_acc = list(0. for i in range(230))
    for i in range(230):
        if class_total[i]!=0:
            class_acc[i] = class_correct[i]/class_total[i]
    # df = pd.DataFrame(class_acc)
    # df.to_excel("rotation_per_class_acc_exp_10_dataset_4alexnet.xlsx")
    # import pdb;pdb.set_trace()
    
    
    # three colummns 1: class total column2: class correct  column3 : class acc
    df = pd.DataFrame({"class_total": class_total, "class_correct": class_correct, "class_acc":class_acc})
    df.to_excel("re_exp/result/per_class_acc_{}_{}.xlsx".format(args.model_path.split('/')[-1], args.test_path.split('/')[-1]))
    
    class_acc7 = class_correct7 / class_total7
    df = pd.DataFrame(class_acc7.cpu().numpy())
    df.to_excel("re_exp/result/per_class_acc7_{}_{}.xlsx".format(args.model_path.split('/')[-1], args.test_path.split('/')[-1]))
    
    
    # save the confusion matrix to excel
    import pandas as pd
    df = pd.DataFrame(confusion_matrix.cpu().numpy())
    df.to_excel("re_exp/result/confusion_matrix_{}_{}.xlsx".format(args.model_path.split('/')[-1], args.test_path.split('/')[-1]))
    
    
    df = pd.DataFrame(confusion_matrix7.cpu().numpy())
    df.to_excel("re_exp/result/confusion_matrix_7_{}_{}.xlsx".format(args.model_path.split('/')[-1], args.test_path.split('/')[-1]))
    
    
    
    
    
        
    
    
    
