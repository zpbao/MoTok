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

banned_scenes = ['scene_000100','scene_000002','scene_000008','scene_000012','scene_000018','scene_000029',
'scene_000038','scene_000040','scene_000043','scene_000044','scene_000049','scene_000050','scene_000053','scene_000063',
'scene_000079','scene_000090','scene_000094','scene_000100','scene_000103','scene_000106','scene_000111','scene_000112',
'scene_000124','scene_000125','scene_000127','scene_000148','scene_000159','scene_000166','scene_000169',
'scene_000170','scene_000171','scene_000187', 'scene_000191','scene_000200','scene_000202','scene_000217',
'scene_000218','scene_000225','scene_000229','scene_000232','scene_000236','scene_000237','scene_000245',
'scene_000249'
]

class PDDataset(Dataset):
    def __init__(self, split='train', root = None, supervision = 'moving'):
        super(PDDataset, self).__init__()       
        self.root_dir = root
        self.files = os.listdir(self.root_dir)
        self.files.sort()
        if split == 'train':
            self.files = self.files[1:]
        elif split == 'eval':
            self.files = self.files[0:1]
        else:
            self.files = self.files
        self.annotation = None 
        if supervision == 'moving':
            self.annotation = 'moving_masks'
        elif supervision == 'all':
            self.annotation = 'ari_masks'
        elif supervision == 'est':
            self.annotation = 'est_masks'
        else:
            raise ValueError("Need to choose from moving masks, all masks, or estimated masks. Or revise the code for customized setting.")
        self.real_files = []
        self.mask_files = []
        self.flow_files = []
        self.depth_files = []
        for f in self.files:
            if f in banned_scenes:
                continue
            for i in [1,5,6,7,8,9]:
                if os.path.exists(os.path.join(self.root_dir,f+'/rgb/camera_0{}'.format(i))):
                    self.real_files.append(f+'/rgb/camera_0{}'.format(i))
                    self.mask_files.append(f+'/{}/camera_0{}'.format(self.annotation, i))
                    self.flow_files.append(f+'/motion_vectors_2d/camera_0{}'.format(i))
                    self.depth_files.append(f+'/depth/camera_0{}'.format(i))
        self.img_transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        path = self.real_files[index]
        mask_path = self.mask_files[index]
        flow_path = self.flow_files[index]
        depth_path = self.depth_files[index]

        all_images = os.listdir(os.path.join(self.root_dir,path))
        all_depths = os.listdir(os.path.join(self.root_dir,depth_path))
        all_images.sort()
        rand_id = random.randint(0,190)
        real_idx = [rand_id + j for j in range(5)]
        ims = []
        masks = []
        flows = []
        depths = []
        mapping = {0:0}
        mapping_count = 1
        for idd in real_idx:
            image = cv2.imread(os.path.join(os.path.join(self.root_dir, path),all_images[idd]))
            mask = cv2.imread(os.path.join(os.path.join(self.root_dir, mask_path),all_images[idd]),-1)
            flow = cv2.imread(os.path.join(os.path.join(self.root_dir, flow_path),all_images[idd]),-1)
            depth = np.load(os.path.join(os.path.join(self.root_dir, depth_path),all_depths[idd]))
            depth = depth['data']
            downsampling_ratio = 0.5
            crop = 128
            width = int(math.ceil(image.shape[1] * downsampling_ratio))
            height = int(math.ceil(image.shape[0] * downsampling_ratio))
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
            image = image[crop:, :, :]
            mask = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
            mask = mask[crop:,:]
            depth = cv2.resize(depth, dim, interpolation = cv2.INTER_NEAREST)
            depth = depth[crop:,:]
            
            r,g,b,a = flow[:,:,0], flow[:,:,1], flow[:,:,2],flow[:,:,3]
            h,w,_ = flow.shape
            dx_i = r+g*256
            dy_i = b+a*256
            flow_x = ((dx_i / 65535.0)*2.0-1.0) * w
            flow_y = ((dy_i / 65535.0)*2.0 - 1.0) * h
            flow = np.zeros((h,w,2), dtype = 'float32')
            flow[:,:,0] += flow_x
            flow[:,:,1] += flow_y
            flow = cv2.resize(flow, dim, interpolation = cv2.INTER_NEAREST)
            flow = flow[crop:,:]
            flow = flow * downsampling_ratio

            values, indices, counts = np.unique(mask, return_inverse=True, return_counts=True)
            for i in range(len(values)):
                if values[i] not in mapping:
                    if counts[i] > 300:
                        mapping[values[i]] = mapping_count
                        mapping_count += 1
            cur_mapping = []
            for value in values:
                if value not in mapping:
                    cur_mapping.append(0)
                else:
                    cur_mapping.append(mapping[value])
            cur_mapping = np.array(cur_mapping)
            _h, _w = mask.shape
            mask = cur_mapping[indices].reshape((_h, _w))

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
