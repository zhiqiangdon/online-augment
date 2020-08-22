#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python main.py  \
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
                        --aug_stn stn_2cycle_diverse \
                        --noise_dim 1 \
                        --linear_size 8 \
                        --adv_weight_stn 0.1 \
                        --div_weight_stn 0.1 \
                        --diversity_weight_stn 0 \
                        --bn_num 2 \
                        --inner_num 2 \
                        --aug_type autoaug_cifar10 \
                        --exp_type aug_stn \

