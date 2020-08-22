#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python main.py  \
                        --data_dir /freespace/local/zt53/data \
                        --exp_dir /freespace/local/zt53/exp \
                        --dataset reduced_cifar10 \
                        --model wresnet28_10 \
                        --batch_size 128 \
                        --epochs 200 \
                        --lr 0.1 \
                        --lr_scheduler cosine \
                        --momentum 0.9 \
                        --weight_decay 5e-4 \
                        --workers 2 \
                        --cutout 16 \
                        --perturb_vae vae_conv_cifar_v1 \
                        --z_dim 8 \
                        --fea_dim 512 \
                        --adv_weight_vae 10 \
                        --div_weight_vae 1e-3 \
                        --aug_stn stn_2cycle_diverse \
                        --noise_dim 1 \
                        --linear_size 8 \
                        --adv_weight_stn 0.1 \
                        --div_weight_stn 0.1 \
                        --diversity_weight_stn 0 \
                        --bn_num 2 \
                        --inner_num 1 \
                        --aug_type autoaug_cifar10 \
                        --exp_type astn_pvae \

#--deform_vae deform_conv_cifar_v1 \
#                        --z_dim_deform 32 \
#                        --fea_dim_deform 512 \
#                        --adv_weight_deform 0.01 \
#                        --div_weight_deform 1 \
#                        --smooth_weight 10 \