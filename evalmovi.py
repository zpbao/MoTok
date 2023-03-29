import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from dataset.datasetMOVIeval import MoviDataset
from models.utils import adjusted_rand_index as ARI 
import torch

from models.model import SlotAttentionAutoEncoder

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--ckpt_path', default='./pre-trained/movi.ckpt', type=str, help='pre-trained model path' )
parser.add_argument('--test_path', default = '/data/MOVI/movi-e/', type = str, help = 'path of MOVI test set')
parser.add_argument('--num_slots', default=24, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--num_tokens', default=64, type=int, help='Number of tokens for VQ-VAE.')

resolution = (128, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    opt = parser.parse_args()
    model_path = opt.ckpt_path

    data_path = opt.test_path
    test_set = MoviDataset(split = 'test', root = data_path)

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.hid_dim, 3, opt.num_tokens, depth = 1).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    print('model load finished!')

    for param in model.module.parameters():
        param.requires_grad = False


    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=8,
                                shuffle=False, num_workers=4, drop_last=False)

    ARIs = []

    for sample in tqdm(test_dataloader):
        image = sample['image']
        image = image.permute(0, 1, 4, 2, 3)
        image = F.interpolate(image, (3, 128,128)).to(device)

        mask_gt = sample['mask'].to(device)
        mask_gt = mask_gt.permute(0,1,4,2,3).squeeze(2)
        mask_gt = F.interpolate(mask_gt.float(), (32,32),mode='nearest').long()
    
        # recon_combined, masks, slots, z_q_x, latents = model(image)
        recon_combined, masks, slots, _, _, latents = model(image)
        K = image.shape[0]
        for i in range(K):
            gt_msk = mask_gt[i]
            pred_msk = masks[i]
            gt_msk = gt_msk.view(24,-1)
            pred_msk = pred_msk.view(24,opt.num_slots,-1).permute(1,0,2)

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
        del image, mask_gt, masks
        # gt_msk = mask_gt[0]
        # pred_msk = masks[0]
        # gt_msk = gt_msk.view(24,-1)
        # pred_msk = pred_msk.view(24,opt.num_slots,-1).permute(1,0,2)

        # gt_msk = gt_msk.view(-1)
        # pred_msk = pred_msk.reshape(opt.num_slots,-1)

        # pred_msk = pred_msk.permute(1,0)
        # gt_msk = F.one_hot(gt_msk)
        # _,n_cat = gt_msk.shape 
        # if n_cat <= 2:
        #     continue
        # gt_msk = gt_msk[:,1:]
        # ari = ARI(gt_msk.unsqueeze(0), pred_msk.unsqueeze(0))
        # ARIs.append(ari)
        # del image, mask_gt, masks
        # print(sum(ARIs) / len(ARIs))
    print(model_path, 'final ARI:',sum(ARIs) / len(ARIs))

if __name__ == '__main__':
    main()