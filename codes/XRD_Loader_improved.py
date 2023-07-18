import torch 
from torch.utils.data import DataLoader, Dataset
import os
from improved import *
import time 
import multiprocessing as mp
class XRD_dataset(Dataset):
    def __init__(self,folder_path, x_step, hkl_info, uvw_info):
        self.folder_path = folder_path
        self.cif_files = []
        self.count_files()
        self.x_step = x_step
        self.hkl_info = hkl_info
        self.uvw_info = uvw_info
        self.pool = mp.Pool(processes=8)
        
    
    
    def count_files(self):
        cif_count = 0
        for path, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith('.cif'):
                    cif_count += 1
                    self.cif_files.append(file)
        self.count = cif_count

    def __getitem__(self,index):
        feature, l7, l230  = cif(self.folder_path, self.cif_files[index], self.x_step, self.hkl_info, self.uvw_info,pool=self.pool)
        return feature, l7, l230
    
    def __len__(self):
        print("Total number of cif files: ", self.count)
        return self.count
    
    
if __name__ == '__main__':
    xrd_dir = "../CIFs_examples"
    hkl_info = hkl(10)
    # print(hkl_info)
    uvw_info = [1,1,1]
    xstep = 0.01
    xrd_dataset = XRD_dataset(xrd_dir, xstep, hkl_info, uvw_info)
    xrd_dataloader = DataLoader(xrd_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    times = []
    start = time.time()
    for i, data in enumerate(xrd_dataloader):
        end = time.time()
        times.append(end-start)
        start = time.time()
    times = np.array(times)
    print('time: ', np.mean(times))
    
    # time:  2.1875834465026855 s 