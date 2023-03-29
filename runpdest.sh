CUDA_VISIBLE_DEVICES=4,5,6,7 python train_pd.py \
--exp_name pdest \
--proj_name motok_release \
--data_path /data/pd_v2/ \
--batch_size 8 \
--supervision est \
--num_slots 45 \
--learning_rate 0.0004 \
--warmup_steps 3000 \
--decay_steps \
--hid_dim 128 \
# --wandb True \