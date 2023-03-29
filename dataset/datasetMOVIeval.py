import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2

import math

class MoviDataset(Dataset):
    def __init__(self, split='train', root = None):
        super(MoviDataset, self).__init__()
        
        self.root_dir = root
        self.split = split
        if split == 'train':
            self.root_dir = os.path.join(self.root_dir, 'train')
        elif split == 'val':
            self.root_dir = os.path.join(self.root_dir, 'val')
        else:
            self.root_dir = os.path.join(self.root_dir, 'test')

        self.files = os.listdir(self.root_dir)
        self.files.sort()

    def __getitem__(self, index):
        path = self.files[index]
        rgb = os.path.join(self.root_dir, os.path.join(path, 'rgb.npy'))
        depth = os.path.join(self.root_dir, os.path.join(path, 'depth.npy'))
        instance = os.path.join(self.root_dir, os.path.join(path, 'segment.npy'))
        flow = os.path.join(self.root_dir, os.path.join(path, 'forward_flow.npy'))

        rgb = np.load(rgb)
        depth = np.load(depth)
        instance = np.load(instance)
        flow = np.load(flow)

        rand_id = 0
        real_idx = [rand_id + j for j in range(24)]
        rgb = rgb[real_idx]
        depth = depth[real_idx]
        instance = instance[real_idx]
        flow = flow[real_idx]

        rgb = torch.Tensor(rgb).float()
        depth = torch.Tensor(depth).float() + 1e-8 
        depth = torch.log(depth + 1)
        instance = torch.Tensor(instance).long()
        flow = torch.Tensor(flow).float()

        rgb = (rgb / 255.0 ) * 2 -1 
        sample = {'image': rgb, 'mask':instance, 'flow': flow, 'depth': depth}
        return sample
            
    
    def __len__(self):
        return len(self.files)
