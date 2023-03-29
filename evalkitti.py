import os
import argparse
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import scipy.optimize
import torch.nn.functional as F
import numpy as np 
import torch 
import math


from dataset.datasetKITTIeval import KITTIDataset
from models.utils import adjusted_rand_index as ARI 
from torchvision.utils import save_image as save_im
from models.model import SlotAttentionAutoEncoder

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--ckpt_path', default='./pre-trained/kitti.ckpt', type=str, help='pre-trained model path' )
parser.add_argument('--test_path', default = '/data/KITTI/test', type = str, help = 'path of MOVI test set')
parser.add_argument('--num_slots', default=45, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--hid_dim', default=128, type=int, help='hidden dimension size')
parser.add_argument('--num_tokens', default=128, type=int, help='Number of tokens for VQ-VAE.')

resolution = (368, 1248)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    opt = parser.parse_args()
    data_path = opt.test_path
    test_set = KITTIDataset(split = 'test', root = data_path)
    model_path = opt.ckpt_path

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.hid_dim, 3, opt.num_tokens).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    print('model load finished!')

    for param in model.module.parameters():
        param.requires_grad = False


    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=8,
                                shuffle=True, num_workers=4, drop_last=False)

    ARIs = []

    for sample in tqdm(test_dataloader):
        image = sample['image'].to(device)
        image = image.unsqueeze(1)
        mask_gt = sample['mask']
        mask_gt = mask_gt.unsqueeze(1)
        
        _, masks, _, _, _, _ = model(image)
        masks = masks.detach().cpu()

        for i in range(8):
            gt_msk = mask_gt[i]
            pred_msk = masks[i]
            gt_msk = gt_msk.view(1,-1)
            pred_msk = pred_msk.view(1,opt.num_slots,-1).permute(1,0,2)

            gt_msk = gt_msk.view(-1)
            pred_msk = pred_msk.reshape(opt.num_slots,-1)


            pred_msk = pred_msk.permute(1,0)
            gt_msk = F.one_hot(gt_msk)
            _,n_cat = gt_msk.shape 
            if n_cat <= 2:
                continue
            gt_msk = gt_msk[:,1:]
            ari = ARI(gt_msk.unsqueeze(0), pred_msk.unsqueeze(0))

            ARIs.append(ari)
        print(sum(ARIs) / len(ARIs))
        del image, masks, mask_gt
    print('final ARI:',sum(ARIs) / len(ARIs))

if __name__ == '__main__':
    main()

