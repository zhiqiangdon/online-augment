#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python main.py  \
                        --data_dir ~/data \
                        --exp_dir ~/exp \
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
                        --deform_vae deform_conv_cifar_v1 \
                        --z_dim_deform 32 \
                        --fea_dim_deform 512 \
                        --adv_weight_deform 0.01 \
                        --div_weight_deform 1 \
                        --smooth_weight 10 \
                        --bn_num 2 \
                        --inner_num 4 \
                        --aug_type basic \
                        --exp_type deform_vae \

