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

from dataset.datasetMOVI import MoviDataset
from  models.model import SlotAttentionAutoEncoder
from models.utils import adjusted_rand_index as ARI 
from torch.nn.utils import clip_grad_norm_
from models.utils import token_loss
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument
# basic configurations
parser.add_argument('--model_dir', default='./tmp/', type=str, help='where to save models' )
parser.add_argument('--sample_dir', default = './samples/', type = str, help = 'where to save the plots')
parser.add_argument('--exp_name', default='', type=str, help='experiment name, used for model saving/plotting/wand ect' )
parser.add_argument('--proj_name', default='my-project', type=str, help='wandb project name' )
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--data_path', default = '/mnt/fsx/pd_v2', type = str, help = 'path of PD dataset')
parser.add_argument('--supervision',  default = 'moving', choices=['moving', 'all', 'est'], help = 'type of supervision, currently available: moving and all')
# model parameters
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_slots', default=45, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_tokens', default=128, type=int, help='Number of tokens for VQ-VAE.')
parser.add_argument('--hid_dim', default=128, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_epochs', default=500, type=int, help='number of workers for loading data')
parser.add_argument('--weight_mask', default = 1.0, type = float, help = 'weight for the mask loss')
parser.add_argument('--weight_token', default = 0.05, type = float, help = 'weight for the token loss')
parser.add_argument('--weight_vq', default = 1.0, type = float, help = 'weight for the vqvae latent loss')
# wandb
parser.add_argument('--wandb', default=False, type = bool)
parser.add_argument('--entity', default='zpbao', type = str, help = 'wandb name')



def main():
    opt = parser.parse_args()

    resolution = (128, 128)

    if opt.wandb:
        import wandb
        wandb.init(project=opt.proj_name, entity=opt.entity, name = opt.exp_name)
    
    if not os.path.exists(opt.model_dir):
        os.mkdir(opt.model_dir)
    if not os.path.exists(opt.sample_dir):
        os.mkdir(opt.sample_dir)
    if not os.path.exists(os.path.join(opt.model_dir, opt.exp_name)):
        os.mkdir(os.path.join(opt.model_dir, opt.exp_name))
    if not os.path.exists(os.path.join(opt.sample_dir, opt.exp_name)):
        os.mkdir(os.path.join(opt.sample_dir, opt.exp_name))

    data_path = opt.data_path
    train_set = MoviDataset(split = 'train', root = data_path)
    test_set = MoviDataset(split = 'eval', root = data_path)

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.hid_dim, 3, opt.num_tokens, depth = 1).to(device)
    model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    bcecriterion = nn.BCELoss()

    params = [{'params': model.parameters()}]

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, drop_last=True)

    optimizer = optim.Adam(params, lr=opt.learning_rate)

    start = time.time()
    step = 0
    print('Model build finished!')
    for epoch in range(opt.num_epochs):
        model.train()

        total_loss = 0
        VQ_loss = 0

        for sample in tqdm(train_dataloader):
            step += 1
        
            if step < opt.warmup_steps:
                learning_rate = opt.learning_rate * (step / opt.warmup_steps)
            else:
                learning_rate = opt.learning_rate

            learning_rate = learning_rate * (opt.decay_rate ** (
                step / opt.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate
            
            image = sample['image'].to(device)
            image = image.permute(0, 1, 4, 2, 3).to(device)
            image = F.interpolate(image, (3, 128,128)).to(device)

            
            recon_combined, masks, slots, z_e_x, z_q_x, latents = model(image)
            recon_combined = recon_combined.view(opt.batch_size,6,3,resolution[0],resolution[1])
            # reconstruction loss
            loss = criterion(recon_combined, image) 
            vq_loss = criterion(z_e_x, z_q_x.detach()) + criterion(z_q_x, z_e_x.detach()) + criterion(slots, z_q_x.detach()) + criterion(z_q_x, slots.detach())
            # mask loss

            
            t_loss = token_loss(model.module.vqvae.codebook.embedding.weight)
            whole_loss = loss + opt.weight_vq*vq_loss + opt.weight_token * t_loss

            optimizer.zero_grad()
            whole_loss.backward(retain_graph = True)
            clip_grad_norm_(model.parameters(),1)
            optimizer.step()

            total_loss += loss.item()
            VQ_loss += vq_loss.item() 
            
            del recon_combined, masks, image, loss, whole_loss, slots, z_q_x, latents, vq_loss, z_e_x
            # break

        total_loss /= len(train_dataloader)
        VQ_loss /= len(train_dataloader)
        

        print ("Epoch: {}, Loss: {}, Loss_vq: {}, Time: {}".format(epoch, total_loss,VQ_loss,
            datetime.timedelta(seconds=time.time() - start)))
        
        sample = test_set[0]
        image = sample['image'].to(device)
        image = image.unsqueeze(0)
        image = image.permute(0, 1, 4, 2, 3)
        image = F.interpolate(image, (3, 128,128)).to(device)

        mask_gt = sample['mask'].to(device)
        mask_gt = mask_gt.permute(0,3,1,2)
        mask_gt = F.interpolate(mask_gt.float(), (32,32)).long()

        recon_combined, masks, slots, _, _, latents = model(image)

        index_mask = masks.argmax(dim = 2)
        index_mask = F.one_hot(index_mask,num_classes = opt.num_slots)
        index_mask = index_mask.permute(0,1,4,2,3)
        masks = masks * index_mask

        image = image[0]
        image = F.interpolate(image, (32,32))
        masks = masks[0]

        recon_combined = recon_combined.detach()
        masks = masks.detach()

        fig, ax = plt.subplots(math.ceil((opt.num_slots+2) / 10), 10, figsize=(45, 5 * math.ceil((opt.num_slots +2)/ 10)))
        for i in range(1):
            image_i = image[i]
            recon_combined_i = recon_combined[i]
            masks_i = masks[i].cpu().numpy()
            image_i = image_i.permute(1,2,0).cpu().numpy()
            image_i = image_i * 0.5 + 0.5
            recon_combined_i = recon_combined_i.permute(1,2,0)
            recon_combined_i = recon_combined_i.cpu().numpy()
            recon_combined_i = recon_combined_i * 0.5 + 0.5
            ax[i,0].imshow(image_i)
            ax[i,0].set_title('Image-f{}'.format(i))
            ax[i,1].imshow(recon_combined_i)
            ax[i,1].set_title('Recon.')
            for j in range(opt.num_slots):               
                ax[(j+2)//10,(j + 2)%10].imshow(image_i)
                ax[(j+2)//10,(j + 2)%10].imshow(masks_i[j], cmap = 'viridis', alpha = 0.6)
                ax[(j+2)//10,(j + 2)%10].set_title('Slot %s' % str(j + 1))
            for j in range(math.ceil((opt.num_slots+2) / 10) * 10):
                ax[(j)//10,(j)%10].grid(False)
                ax[(j)//10,(j)%10].axis('off')
        eval_name = os.path.join(opt.sample_dir,opt.exp_name,'epoch_{}_slot.png'.format(epoch))
        fig.savefig(eval_name)
        plt.close(fig)

        latents = F.one_hot(latents, num_classes = opt.num_tokens)
        l_sum = latents.sum(dim = (0,1,2))
        _, l_idx = torch.topk(l_sum, opt.num_slots)
        latents = latents[:,:,:,l_idx]
        latents = latents.detach()

        fig, ax = plt.subplots(math.ceil((opt.num_slots+2) / 10), 10, figsize=(45, 5 * math.ceil((opt.num_slots +2)/ 10)))
        for i in range(1):
            image_i = image[i]
            recon_combined_i = recon_combined[i]
            masks_i = latents[i].cpu().numpy()
            image_i = image_i.permute(1,2,0).cpu().numpy()
            image_i = image_i * 0.5 + 0.5
            recon_combined_i = recon_combined_i.permute(1,2,0)
            recon_combined_i = recon_combined_i.cpu().numpy()
            recon_combined_i = recon_combined_i * 0.5 + 0.5
            ax[i,0].imshow(image_i)
            ax[i,0].set_title('Image-f{}'.format(i))
            ax[i,1].imshow(recon_combined_i)
            ax[i,1].set_title('Recon.')
            for j in range(opt.num_slots):               
                ax[(j+2)//10,(j + 2)%10].imshow(image_i)
                ax[(j+2)//10,(j + 2)%10].imshow(masks_i[:,:,j], cmap = 'viridis', alpha = 0.6)
                ax[(j+2)//10,(j + 2)%10].set_title('Token %s' % str(j + 1))
            for j in range(math.ceil((opt.num_slots+2) / 10) * 10):
                ax[(j)//10,(j)%10].grid(False)
                ax[(j)//10,(j)%10].axis('off')
        eval_name = os.path.join(opt.sample_dir,opt.exp_name,'epoch_{}_vqtoken.png'.format(epoch))
        fig.savefig(eval_name)
        plt.close(fig)

        gt_msk = mask_gt.detach()
        pred_msk = masks
        gt_msk = gt_msk.view(6,-1)
        pred_msk = pred_msk.view(6,24,-1).permute(1,0,2)

        gt_msk = gt_msk.view(-1)
        pred_msk = pred_msk.reshape(24,-1)

        idx = gt_msk>0
        gt_msk = gt_msk[idx]

        pred_msk = pred_msk[:,idx]
        pred_msk = pred_msk.permute(1,0)
        gt_msk = F.one_hot(gt_msk)
        ari = ARI(gt_msk.unsqueeze(0), pred_msk.unsqueeze(0))

        if opt.wandb:
            wandb.log({"recon_loss": total_loss,  "vq_loss": VQ_loss, 'test_ari': ari})
        

        del masks, recon_combined, image, slots, latents 
        
        if not epoch % 10:
            torch.save({
                'model_state_dict': model.state_dict(),
                }, os.path.join(opt.model_dir, opt.exp_name, 'epoch_{}.ckpt'.format(epoch))
                )

if __name__ == '__main__':
    main()
