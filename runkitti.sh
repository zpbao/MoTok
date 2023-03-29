CUDA_VISIBLE_DEVICES=4,5,6,7 python train_kitti.py \
--exp_name kitti \
--proj_name motok_release \
--data_path /data/KITTI/data/ \
--batch_size 4 \
--num_slots 45 \
--learning_rate 0.0004 \
--warmup_steps 3000 \
--hid_dim 128 \
# --wandb True \