from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
import torch
import numpy as np
import pandas as pd
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)


preprocess = transforms.Compose([
    # transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=90,fill=255),
    transforms.ToTensor(),
])

def default_loader(path):
    # try:
    img_pil =  Image.open(path).convert('RGB')
    img_pil = img_pil.resize((224,224))
    # import pdb; pdb.set_trace()
    # save this image
    # img_pil.save("test.png")
    img_tensor = preprocess(img_pil)
    
    return img_tensor


import re




#当然出来的时候已经全都变成了tensor
class Dataset(Dataset):
    def __init__(self, root_path, loader=default_loader,fmt=True):
        #定义好 image 的路径
        self.root_path = root_path
        self.img_files = os.listdir(root_path)
        # remove the files that are not images
        self.img_files = [x for x in self.img_files if x.endswith('.png')]
        self.loader = loader
        
    
    def __getitem__(self, index):
        fn = self.img_files[index]
        # example fn:  train_icsd_240253_u1_v0_w0_tet_cub.png
        # I want to extract the h,k,l values: 1,0,0 (followed by u,v,w)
        hkl_values = re.findall(r'_u(\d+)_v(\d+)_w(\d+)_', fn)
        
        # output is [('1', '0', '0')], transform to integers
        hkl_values = [int(x) for x in hkl_values[0]]
        h, k, l = hkl_values[0], hkl_values[1], hkl_values[2]
        img = self.loader(os.path.join(self.root_path, fn))
        return img, h, k, l, fn
      
	    
            # import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()
            # return self.__getitem__(index+1)
        

    def __len__(self):
        return len(self.img_files)
    
    
class ComplexDataset(Dataset):
    def __init__(self, root_path, img_path, label_path,loader=default_loader,fmt=False):
        #定义好 image 的路径
        self.root_path = root_path
        self.img_path = img_path
        self.label_path = label_path
        self.loader = loader
        self.img_files = pd.read_excel(self.img_path, header=None).to_numpy().flatten() # read xlsx file
        self.label_file = pd.read_excel(self.label_path, header=None).to_numpy() # read xlsx file
        self.unions = {}
        self.unions_labels = {}
        # (filename, label) -> (filename: label) 
        if not fmt:
            self.labels = {x[0]:x[1] for x in self.label_file}
        else:
            self.labels = {x[0].split('.')[0]:x[1] for x in self.label_file}
        
        for key_name in self.labels:
            key = key_name[:-9]
            if key not in self.unions:
                self.unions[key] = []
                self.unions_labels[key] = self.labels[key_name]-1
            else:
                assert self.unions_labels[key] == self.labels[key_name]-1

        # import pdb; pdb.set_trace()

    
    def __getitem__(self, index):
        fn = self.img_files[index]
        img = self.loader(os.path.join(self.root_path, fn))
        fn_remote_suffix = fn.split('.')[0]
        target = self.labels[fn_remote_suffix] - 1 
        union_name = fn_remote_suffix[:-9]
        return union_name, img, target

    def __len__(self):
        return len(self.img_files)
    
class Balanced_ComplexDataset(Dataset):
    def __init__(self, root_path, img_path, label_path,loader=default_loader,fmt=False):
        #定义好 image 的路径
        self.root_path = root_path
        self.img_path = img_path
        self.label_path = label_path
        self.loader = loader
        self.img_files = pd.read_excel(self.img_path, header=None).to_numpy().flatten() # read xlsx file
        self.label_file = pd.read_excel(self.label_path, header=None).to_numpy() # read xlsx file
        self.unions = {}
        self.unions_labels = {}
        
        # (filename, label) -> (filename: label) 
        if not fmt:
            self.labels = {x[0]:x[1] for x in self.label_file}
        else:
            self.labels = {x[0].split('.')[0]:x[1] for x in self.label_file}
        
        for key_name in self.labels:
            key = key_name[:-9]
            if key not in self.unions:
                self.unions[key] = []
                self.unions_labels[key] = self.labels[key_name]-1
            else:
                assert self.unions_labels[key] == self.labels[key_name]-1
        self.class_to_imgs = self.func_class_to_imgs()
        # import pdb; pdb.set_trace()

    def func_class_to_imgs(self):
        # group the images by their classes
        class_to_imgs = {}
        for img_name, label in self.labels.items():
            if label not in class_to_imgs:
                class_to_imgs[label] = []
            class_to_imgs[label].append(img_name)
        return class_to_imgs
    
    
    def __getitem__(self, index):
        # sample a class based on index (should random but depend on index), then sample an image from this class
        class_index = index % len(self.class_to_imgs)
        class_label = list(self.class_to_imgs.keys())[class_index]
        # img_name = random.choice(self.class_to_imgs[class_label])
        img_name = self.class_to_imgs[class_label][index % len(self.class_to_imgs[class_label])]
        img = self.loader(os.path.join(self.root_path, '2D', img_name+'.png'))
        target = self.labels[img_name] - 1
        union_name = img_name[:-9]
        return union_name, img, target

    def __len__(self):
        return len(self.img_files)   
    
    
if __name__ == "__main__":
    root_path = "/scratch/zzh136/xrd_2d/exp6/output"
    img_path = "/scratch/zzh136/xrd_2d/exp6/output/val_files.xlsx"
    label_path = os.path.join(root_path, "label_train_eval_test_4zone_001_010_100_111.xlsx")
    dataset = Dataset(root_path=root_path, img_path = img_path, label_path = label_path, fmt=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i, (img, label) in enumerate(dataloader):
        print(i, img.shape, label.shape)
        
    
