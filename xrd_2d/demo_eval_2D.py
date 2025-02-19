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
from dataloader import Dataset
from torch.utils.data import  DataLoader
import wandb
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# wandb.init(project="xrd_2d_alexnet")

if __name__ == "__main__":
    root_path = "/scratch/zzh136/xrd_2d/output_random"
    train_path = "/scratch/zzh136/xrd_2d/output_random/train_files.xlsx"
    val_path = "/scratch/zzh136/xrd_2d/output_random/val_files.xlsx"
    
    print('begin initialization')
    label_path = os.path.join(root_path, "labels.xlsx")
    # train_dataset = Dataset(root_path = root_path, img_path = train_path, label_path = label_path)
    val_dataset = Dataset(root_path = root_path, img_path = val_path, label_path = label_path)
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    print('build val dataset ok')
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=8)
    print('build val loader ok')
    
    model = models.alexnet(pretrained=False, num_classes=230)
    optimazier = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("/scratch/zzh136/xrd_2d/exp5/output/ckpt/vanilla_alexnet.pth"))
    # 111: /scratch/zzh136/xrd_2d/output111/ckpt/vanilla_alexnet.pth on 001: 0.0533    on 111: 0.7838
    # 001: /scratch/zzh136/xrd_2d/output/ckpt/vanilla_alexnet.pth    on 001: 0.7496    on 111: 0.0493  on 100: 0.2681 on 010
    model = model.to(device)
    epochs = 100
    batch_size = 128
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # model_save_path = os.path.join(root_path, "ckpt", "vanilla_alexnet.pth")
    # if not os.path.exists(os.path.dirname(model_save_path)):
    #     os.makedirs(os.path.dirname(model_save_path))
    
    # statistic the accuracy for each class
    class_correct = list(0. for i in range(230))
    class_total = list(0. for i in range(230))
    class_correct = np.array(class_correct)
    class_total = np.array(class_total)
    class_correct = torch.tensor(class_correct)
    class_total = torch.tensor(class_total)
    class_correct = class_correct.to(device)
    class_total = class_total.to(device)
    model.eval()
    running_loss = 0.0
    running_correct = 0.0
    N = 0
    confusion_matrix = torch.zeros(230, 230)
    for i, data in tqdm(enumerate(val_loader, 0)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        N += inputs.size(0)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_correct += torch.sum(preds == labels.data)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        # tqdm.write(f"Val Epoch {epoch}/{epochs}, batch {i}/{len(val_loader)}, loss: {loss.item()/(i+1)}, acc: {running_correct.double()/N}")
    val_loss = running_loss / len(val_loader)
    val_acc = running_correct.double() / N
    # class_correct = class_correct.cpu().numpy()
    # class_total = class_total.cpu().numpy()
    # class_acc = class_correct / class_total
    # wandb.log({"val_loss": val_loss, "val_acc": val_acc})
    print(f" val_loss: {val_loss}, val_acc: {val_acc}")
    
    plt.figure(figsize=(15, 7))
    per_class_accuracy = confusion_matrix.diag()/confusion_matrix.sum(1)
    plt.bar(range(230), per_class_accuracy.cpu().numpy())
    plt.savefig("class_accIdea1_random.png")
    
    # save the confusion matrix to excel
    # import pandas as pd
    # df = pd.DataFrame(confusion_matrix.cpu().numpy())
    # df.to_excel("confusion_matrix 1001_010.xlsx")
    # plot the confusion matrix  # 111 001 -> 001_100
    # import seaborn as sns
    # sns.heatmap(confusion_matrix.cpu().numpy())
    # plt.savefig("confusion_matrix_001_111.png")
    # plot the per-class accuracy by scatter plot
    
    colors = np.random.rand(230)
    # area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
    plt.scatter(range(230), per_class_accuracy.cpu().numpy(), c=colors)
    plt.savefig("class_acc_scatter_Idea1_random_scatter.png")
    
    # bar plot the class_acc to show the range
    # plt.hist(class_acc, bins=20)
    # plt.savefig("class_acc.png")
    
    
    
    # torch.save(model.state_dict(), model_save_path)
    # print("Model saved to ", model_save_path)
        
    
    
    