import torch 
from torch.utils.data import DataLoader, Dataset
import os
from direct_cif_reader import *
from func_hkl import *
import time 
from XRD_Loader import *
from tqdm import tqdm

# define a MLP, the input is 8500 and output is 7
class MLP(torch.nn.Module):
    def __init__(self, input_nums = 8500, output_nums = 7):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_nums, 1000)
        self.fc2 = torch.nn.Linear(1000, 100)
        self.fc3 = torch.nn.Linear(100, output_nums)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        # self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    


if __name__ == '__main__':
    xrd_dir = "../CIFs_examples"
    hkl_info = hkl(10)
    # print(hkl_info)
    uvw_info = [1,1,1]
    xstep = 0.01
    xrd_dataset = XRD_dataset(xrd_dir, xstep, hkl_info, uvw_info)
    xrd_dataloader = DataLoader(xrd_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    net = MLP()
    net.train()
    net.cuda()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for _ in tqdm(range(10), desc='epoch'):
        acc_a = 0.0
        loss_a = 0.0
        n = 0
        for i, data in enumerate(xrd_dataloader):
            feature, l7, l230 = data
            # import pdb; pdb.set_trace()
            feature = feature.squeeze(1).cuda().float()
            l7 = l7.cuda()
            l230 = l230.cuda()
            optimizer.zero_grad()
            y = net(feature)
            loss = loss_func(y, l7)
            loss.backward()
            optimizer.step()
            acc = (y.argmax(dim=1) == l7).float().cpu().sum()
            n += len(l7)
            acc_a += acc
            loss_a += loss.detach().cpu().item()
        tqdm.write('loss: {}, acc: {}'.format(loss_a/n, acc_a/n))
        
        
        