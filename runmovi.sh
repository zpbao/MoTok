CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_movi.py \
--exp_name movi \
--proj_name motok_release \
--data_path /data/MOVI/movi-e/ \
--batch_size 64 \
--num_slots 24 \
--learning_rate 0.0005 \
--warmup_steps 3000 \
--hid_dim 64 \
--num_tokens 64 \
--decay_steps 50000 \
# --wandb True \