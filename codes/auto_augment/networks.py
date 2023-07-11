import torch

from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
# from torchvision import models




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



def get_model():
    name = conf['type']

    model = MLP()

    model = model.cuda()
    model = DataParallel(model)
    cudnn.benchmark = True
    return model

def get_model_np(conf, num_class=10):
    name = conf['type']

    model = MLP()
    model = model.cuda()
    #model = DataParallel(model)
    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'xrd7':7,
        'xrd230':230,
    }[dataset]