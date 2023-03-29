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

from dataset.datasetKITTIeval import KITTIDataset
from  models.model import SlotAttentionAutoEncoder

import math

import cv2
from matplotlib.patches import Polygon
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resolution = (368, 1248)

evalname = 'kittiplot'

if not os.path.exists('./{}/'.format(evalname)):
    os.mkdir('./{}/'.format(evalname))

model_path = './pre-trained/kitti.ckpt'

data_path = '/data/KITTI/test'
test_set = KITTIDataset(split = 'test', root = data_path)
    
model = SlotAttentionAutoEncoder(resolution, 45, 128, 3, 128).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(model_path)['model_state_dict'])

cmap = plt.get_cmap('rainbow')
tk = 10
colors = [cmap(i) for i in np.linspace(0, 1, tk)]
colors_rain = [cmap(i) for i in np.linspace(0, 1, 128)]
colors_rain = np.array(colors_rain)
np.random.seed(6)
np.random.shuffle(colors_rain)
np.random.shuffle(colors)


print('model load finished!')

for param in model.module.parameters():
    param.requires_grad = False

for k in range(len(test_set)):
    print(k)
    sample = test_set[k]
    image = sample['image'].to(device)
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)
    image = image.repeat(1,5,1,1,1)

    recon_combined, masks, slots, _,_, latents = model(image)

    index_mask = masks.argmax(dim = 2)
    index_mask = F.one_hot(index_mask,num_classes = 45)
    index_mask = index_mask.permute(0,1,4,2,3)
    masks = masks * index_mask

    image = F.interpolate(image, (3,92,312))
    masks = masks.detach()
    masks =  masks[0]
    image = image[0]

    latents = latents.detach().cpu().numpy()
    
    for j in range(1):
        # cur_ari = (ari[j]>0).astype(float)
        image_j = image[j].permute(1,2,0).cpu().numpy()
        image_j = image_j * 0.5 + 0.5

        masks_j = masks[j]
        scores = masks_j.sum(dim = 1).sum(dim = 1) / ((masks_j>0).float().sum(dim = 1).sum(dim = 1) + 0.000001)
        _, idx = torch.topk(scores, tk)
        
        masks_j = masks_j[idx,:,:]
        masks_j = masks_j.cpu().numpy()

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        image_j = image_j[:,:,-1::-1]
        ax.imshow(image_j, alpha = 1)
        
        for seg in range(tk):
            msk = masks_j[seg]
            threshold = 0

            e = (msk > threshold).astype('uint8')
            contour, hier = cv2.findContours(
                    e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cmax = None
            for c in contour:
                if cmax is None:
                    cmax = c
                if len(c) > len(cmax):
                    cmax = c
            if (msk > 0).sum() > 2000:
                continue
            if cmax is None:
                print(j, seg)
                continue
            else:
                polygon = Polygon(
                    cmax.reshape((-1, 2)),
                    fill=True, facecolor=colors[seg],
                    edgecolor='r', linewidth=0.0,
                    alpha=0.65)
            ax.add_patch(polygon)
        fig.savefig('./{}/scene-{}-frame-{}.png'.format(evalname,k,j))
        plt.close(fig)

        latents_j = latents[j]
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        tmp = latents_j.reshape((92*312,))
        tmp = colors_rain[tmp]
        tmp = tmp.reshape((92,312,4))
        ax.imshow(tmp, alpha = 1)
        fig.savefig('./{}/token-{}-frame-{}.png'.format(evalname,k,j))
        plt.close(fig)
