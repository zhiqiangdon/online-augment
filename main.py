import itertools
import json
import logging
import math
import os
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel

# from tqdm import tqdm

# from common import get_logger

# logger = get_logger('Augmentation')
# logger.setLevel(logging.INFO)

# manualSeed = 1
# import numpy as np
# np.random.seed(manualSeed)
# import random
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)
# # # if you are suing GPU
# torch.cuda.manual_seed(manualSeed)
# torch.cuda.manual_seed_all(manualSeed)
#
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

import argparse
parser = argparse.ArgumentParser(description='OnlineAugment')
# parser = ConfigArgumentParser(conflict_handler='resolve')
parser.add_argument('--epochs', default=160, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')  # 0.1
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')  # 0.9
parser.add_argument('--weight_decay', '-wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
parser.add_argument('--dataset', default='cifar100', type=str, help='cifar100, mnist')
parser.add_argument('--exp_dir', default='/dresden/gpu2/zt53-2/zt53/exp2', type=str, help='exp dir')
parser.add_argument('--exp_id', default='test', type=str, help='exp id')
parser.add_argument('--data_dir', default='/dresden/gpu2/zt53-2/zt53/data', type=str, help='exp id')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--cutout', type=int, default=0)
parser.add_argument('--save', type=str, default='test.pth')
parser.add_argument('--lr_scheduler', type=str, default=None)
parser.add_argument('--warmup_epoch', type=int, default=0)
parser.add_argument('--warmup_multiplier', type=int, default=0)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--aug_type', type=str, default='basic')
parser.add_argument('--pyramidnet_depth', type=int, default=272)
parser.add_argument('--pyramidnet_alpha', type=int, default=200)
parser.add_argument('--grad_clip', type=float, default=5.)
parser.add_argument('--only-eval', action='store_true')
parser.add_argument('--exp_type', default='baseline', type=str, help='experimental type')
parser.add_argument('--bn_num', type=int, default=1)
parser.add_argument('--aug_net_lr', default=1e-3, type=float, help='aug net lr')
parser.add_argument('--aug_net_weight_decay', default=1e-4, type=float, help='aug net weight decay')
parser.add_argument('--adam_beta1', default=0.5, type=float, help='beta1 para in the adam optimizer')
parser.add_argument('--perturb_vae', default=None, type=str, help='vae_conv_cifar_v1')
parser.add_argument('--adv_weight_vae', default=None, type=float, help='weight of adversarial loss')
parser.add_argument('--div_weight_vae', default=None, type=float, help='weight for divergence loss')
parser.add_argument('--z_dim', default=16, type=int, help='latent vector dimension in vae')
parser.add_argument('--fea_dim', default=512, type=int, help='feature dimension in vae')
parser.add_argument('--aug_stn', default=None, type=str, help='stn_2cycle_diverse')
parser.add_argument('--adv_weight_stn', default=None, type=float, help='weight of adversarial loss')
parser.add_argument('--div_weight_stn', default=None, type=float, help='weight for divergence loss')
parser.add_argument('--diversity_weight_stn', default=None, type=float, help='weight for stn diversity loss')
parser.add_argument('--noise_dim', default=1, type=int, help='input dimension for stn input')
parser.add_argument('--linear_size', default=None, type=int, help='fc size in stn localization net')
parser.add_argument('--deform_vae', default=None, type=str, help='vae_deform_conv_cifar')
parser.add_argument('--adv_weight_deform', default=None, type=float, help='weight of adversarial loss')
parser.add_argument('--div_weight_deform', default=None, type=float, help='weight for divergence loss')
parser.add_argument('--smooth_weight', default=None, type=float, help='weight for smooth loss')
parser.add_argument('--z_dim_deform', default=16, type=int, help='latent vector dimension in deformation vae')
parser.add_argument('--fea_dim_deform', default=512, type=int, help='feature dimension in deformation vae')
parser.add_argument('--sample_num', default=None, type=int, help='sample num')
parser.add_argument('--iter_grad_type', default=None, type=str, help='PGD/GD/IFGSM')
parser.add_argument('--epsilon', default=None, type=int, help='ball size of iter grad attack')
parser.add_argument('--alpha', default=None, type=int, help='learning rate of iter grad attack')
parser.add_argument('--scale', default=None, type=float, help='scale factor of image')
parser.add_argument('--decay_type', default=None, type=str, help='no_bn')
parser.add_argument('--img_res', default=None, type=int, help='image resolution for noise or deformation generator')
parser.add_argument('--inner_num', default=None, type=int, help='number of updating aug net')
parser.add_argument('--inner_type', default=None, type=str, help='aug/both, the type of inner update')
args = parser.parse_args()

def main():
    # assert not (args.horovod and args.only_eval), 'can not use horovod when evaluation mode is enabled.'
    # assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    # if not args.only_eval:
    #     if args.save:
    #         logger.info('checkpoint will be saved at %s' % args.save)
    #     else:
    #         logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')

    import time
    t = time.time()

    aug_num = 0
    if args.perturb_vae: aug_num += 1
    if args.aug_stn: aug_num += 1
    if args.deform_vae: aug_num += 1

    generate_exp_id(aug_num)

    if 'baseline' in args.exp_type:
        from train_baseline import train_and_validate
        train_and_validate(args)
    elif aug_num > 1:
        from train_comb import train_and_validate
        train_and_validate(args)
    elif args.perturb_vae:
        from train_perturb_vae import train_and_validate
        train_and_validate(args)
    elif args.aug_stn:
        from train_aug_stn import train_and_validate
        train_and_validate(args)
    elif args.deform_vae:
        from train_deform_vae import train_and_validate
        train_and_validate(args)
    else:
        raise Exception('unkown exp_type: {}'.format(args.exp_type))
    elapsed = time.time() - t
    print('elapsed time: {:.3f} Hours'.format(elapsed / 3600.))
    # logger.info('done.')
    # logger.info('model: {}'.format(args.model))
    # logger.info('augmentation: {}'.format(args.aug_type))
    # # logger.info('\n' + json.dumps(result, indent=4))
    # logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    # logger.info('top1 error in testset: {.3f}'.format(100. - test_acc))
    # logger.info(args.save)

def generate_exp_id(aug_num):
    exp_id = ''
    exp_id += 'dataset^{}^-'.format(args.dataset)
    exp_id += 'model^{}^-'.format(args.model)
    # exp_id += 'bs^{}^-'.format(args.batch_size)
    exp_id += 'es^{}^-'.format(args.epochs)
    # exp_id += 'wd^{}^-'.format(args.weight_decay)
    exp_id += 'lr^{}^-'.format(args.lr)
    # exp_id += 'lr_sch^{}^-'.format(args.lr_scheduler)
    # if args.aug_type:
    exp_id += 'aug_type^{}^-'.format(args.aug_type)
    if args.perturb_vae:
        # exp_id += 'perturb_vae^{}^-'.format(args.perturb_vae)
        if aug_num <= 1:
            exp_id += 'z_dim^{}^-'.format(args.z_dim)
            exp_id += 'fea_dim^{}^-'.format(args.fea_dim)
            exp_id += 'adv_wt_vae^{}^-'.format(args.adv_weight_vae)
            exp_id += 'recon_wt_vae^{}^-'.format(args.div_weight_vae)
        else:
            exp_id += 'texture-^{}^-^{}^-^{}^-^{}^-'\
                .format(args.z_dim, args.fea_dim, args.adv_weight_vae,
                        args.div_weight_vae)
    if args.aug_stn:
        # exp_id += 'aug_stn^{}^-'.format(args.aug_stn)
        if aug_num <= 1:
            exp_id += 'noise_dim^{}^-'.format(args.noise_dim)
            exp_id += 'l_size^{}^-'.format(args.linear_size)
            exp_id += 'adv_stn^{}^-'.format(args.adv_weight_stn)
            exp_id += 'recon_stn^{}^-'.format(args.div_weight_stn)
            exp_id += 'diverse_stn^{}^-'.format(args.diversity_weight_stn)
        else:
            exp_id += 'stn-^{}^-^{}^-^{}^-^{}^-^{}^-'\
                .format(args.noise_dim, args.linear_size,
                        args.adv_weight_stn, args.div_weight_stn,
                        args.diversity_weight_stn)
    if args.deform_vae:
        # exp_id += 'deform_vae^{}^-'.format(args.deform_vae)
        if aug_num <= 1:
            exp_id += 'z_dim_de^{}^-'.format(args.z_dim_deform)
            exp_id += 'fea_dim_de^{}^-'.format(args.fea_dim_deform)
            exp_id += 'adv_wt_de^{}^-'.format(args.adv_weight_deform)
            exp_id += 'recon_wt_de^{}^-'.format(args.div_weight_deform)
            exp_id += 'smooth_wt^{}^-'.format(args.smooth_weight)
        else:
            exp_id += 'deform-^{}^-^{}^-^{}^-^{}^-^{}^-' \
                .format(args.z_dim_deform, args.fea_dim_deform,
                        args.adv_weight_deform, args.div_weight_deform,
                        args.smooth_weight)
    if args.inner_type is not None:
        exp_id += 'inner_type^{}^-'.format(args.inner_type)
    if args.inner_num is not None:
        exp_id += 'inner_num^{}^-'.format(args.inner_num)

    exp_id += 'exp_type^{}^-'.format(args.exp_type)
    exp_id += 'bn_num^{}^-'.format(args.bn_num)
    if args.sample_num:
        exp_id += 's^{}^-'.format(args.sample_num)
    assert args.bn_num >= 1

    # if args.scale:
    #     exp_id += 'scale^{}^-'.format(args.scale)

    args.exp_id = exp_id
    print('exp_id: {}'.format(args.exp_id))
    # os.mkdir(config.exp_id)
    # exit()

if __name__ == '__main__':
    # print('type of config: {}'.format(type(config)))
    # print('config.z_dim: {}'.format(config.z_dim))
    # config.z_dim = 1
    # print('config.z_dim: {}'.format(config.z_dim))
    # exit()
    main()