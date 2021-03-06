#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python main.py  \
                        --data_dir ~/data \
                        --exp_dir ~/exp \
                        --dataset cifar10 \
                        --model wresnet28_10 \
                        --batch_size 128 \
                        --epochs 200 \
                        --lr 0.1 \
                        --lr_scheduler cosine \
                        --momentum 0.9 \
                        --weight_decay 5e-4 \
                        --workers 2 \
                        --cutout 16 \
                        --aug_type autoaug_cifar10 \
                        --exp_type baseline \
