import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

from torchvision.io import read_video
import random
import cv2
import math

resolution = (1248,368)
dresolution = (312,92)

class KITTIDataset(Dataset):
    def __init__(self, split='train', root = None):
        super(KITTIDataset, self).__init__()
        self.resolution = resolution
        self.root_dir = root
        self.files = os.listdir(self.root_dir)
        self.files.sort()
        self.files = self.files[:151]
        if split == 'train':
            self.files = self.files[5:]
        else:
            self.files = self.files[0:5]
        self.real_files = []
        self.mask_files = []
        self.flow_files = []
        self.depth_files = []
        for f in self.files:
            for i in ['image_02','image_03']:
                if os.path.exists(os.path.join(self.root_dir,f+'/{}/'.format(i))):
                    self.real_files.append(f+'/{}/data'.format(i))
                    self.mask_files.append(f+'/{}/raft_seg'.format(i))
                    self.flow_files.append(f+'/{}/raft_flow'.format(i))
                    self.depth_files.append(f+'/{}/depth'.format(i))
        self.img_transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        path = self.real_files[index]
        mask_path = self.mask_files[index]
        flow_path = self.flow_files[index]
        depth_path = self.depth_files[index]

        all_images = os.listdir(os.path.join(self.root_dir,path))
        all_depths = os.listdir(os.path.join(self.root_dir,depth_path))
        all_flows = os.listdir(os.path.join(self.root_dir,flow_path))
        all_images.sort()
        all_depths.sort()
        all_flows.sort()
        N = len(all_images)
        rand_id = random.randint(0,N-10)
        # rand_id = 0
        real_idx = [rand_id + j for j in range(5)]
        ims = []
        masks = []
        flows = []
        depths = []
        for idd in real_idx:
            image = cv2.imread(os.path.join(os.path.join(self.root_dir, path),all_images[idd]))
            mask = cv2.imread(os.path.join(os.path.join(self.root_dir, mask_path),all_images[idd]),-1)
            flow = np.load(os.path.join(os.path.join(self.root_dir, flow_path),all_flows[idd]))
            depth = np.load(os.path.join(os.path.join(self.root_dir, depth_path),all_depths[idd]))
            depth = depth['depth']

            image = cv2.resize(image, resolution, interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dresolution, interpolation = cv2.INTER_NEAREST)
            depth = cv2.resize(depth, dresolution, interpolation = cv2.INTER_NEAREST)
            flow = cv2.resize(flow, dresolution, interpolation = cv2.INTER_NEAREST)

            mask = torch.Tensor(mask).long()
            image = torch.Tensor(image).float()
            flow = torch.Tensor(flow).float()
            flow = flow.permute(2,0,1)
            depth = torch.Tensor(depth).float()
            image = image / 255.0
            image = image.permute(2,0,1)
            depth = depth.clip_(0,999)
            depth = torch.log(1+depth)
            image = self.img_transform(image)
            ims.append(image)
            masks.append(mask)
            flows.append(flow)
            depths.append(depth)
        ims = torch.stack(ims)
        masks = torch.stack(masks)
        depths = torch.stack(depths)
        flows = torch.stack(flows)
        sample = {'image': ims, 'mask':masks, 'depth': depths, 'flow': flows}
        return sample
            
    
    def __len__(self):
        return len(self.real_files)
