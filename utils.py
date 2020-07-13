import numpy as np
import os
import math
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
from time import strftime
from collections import defaultdict
from six import iteritems
import json
import shutil
# import torch.optim as optim
import models as models
from torchvision.utils import make_grid, save_image
from models.normalize_mean_std import NormalizeByChannelMeanStd

def get_log_dir_path(root_path, run_name):
    """
    Creates log dir of format e.g.:
        experiments/log/2017_01_01/run_name_12_00_00/
    """
    date_stamp = strftime("%Y_%m_%d")
    time_stamp = strftime("%H_%M_%S")

    # Group logs by day first
    log_path = os.path.join(root_path, date_stamp)

    # Then, group by run_name and hour + min + sec to avoid duplicates
    log_path = os.path.join(log_path, "_".join([run_name, time_stamp]))
    return log_path

def get_lr_cosine_decay(config, epoch):
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / config.epochs))
    return config.update_lr * cosine_decay

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def error(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


def get_model_name(model):
    if type(model) == nn.DataParallel:
        return model.module.__class__.__name__
    else:
        return model.__class__.__name__


def save_checkpoint(model, state, is_best, save_dir):
    checkpoint_path = os.path.join(save_dir, '{}_last_ckpt'.format(get_model_name(model)))
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(save_dir, '{}_best_ckpt'.format(get_model_name(model))))


def load_checkpoint( config, model, optimizer = None, load_best = False ):
    if load_best:
        checkpoint_path = os.path.join(config.cp_root, '{}_best_model'.format(get_model_name(model)))
    else:
        # checkpoint_path = os.path.join(config.cp_root, '{}_checkpoint'.format(getModelName(model)))
        checkpoint_path = '{}_checkpoint'.format(get_model_name(model))
    if os.path.isfile(checkpoint_path):
        print('=> loading checkpoint "{}"'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    return best_acc, start_epoch


# save visualized images for img input vae
def save_vis_imgs(model, imgs_vis, writer, epoch, vis_dir, config):
    imgs_vis_aug = model.aug_net(imgs_vis).cpu()
    # print('imgs_vis_aug shape: {}'.format(imgs_vis_aug.size()))
    grid = make_grid(imgs_vis_aug, nrow=int(math.sqrt(imgs_vis_aug.size(0))),
                     normalize=config.vis_nrm, padding=1, pad_value=1)
    # print('imgs_vis_aug shape: {}'.format(grid.size()))
    # writer.add_image('imgs_aug_{}'.format(epoch), grid, 0)
    writer.add_image('imgs_aug', grid, epoch)
    save_image(grid, os.path.join(vis_dir,
                                  'aug_imgs_{}'.format(epoch) + '.png'))


# save visualized images for img input vae
def save_vis_imgs_vae(model, imgs_vis, writer, epoch, vis_dir, config):
    imgs_vis_aug = model.vae(imgs_vis).cpu()
    # print('imgs_vis_aug shape: {}'.format(imgs_vis_aug.size()))
    grid = make_grid(imgs_vis_aug, nrow=int(math.sqrt(imgs_vis_aug.size(0))),
                     normalize=config.vis_nrm, padding=1, pad_value=1)
    # print('imgs_vis_aug shape: {}'.format(grid.size()))
    # writer.add_image('imgs_aug_{}'.format(epoch), grid, 0)
    writer.add_image('imgs_vae', grid, epoch)
    save_image(grid, os.path.join(vis_dir,
                                  'vae_imgs_{}'.format(epoch) + '.png'))


# save visualized images for noise input stn with double cycle loss
def save_vis_imgs_stn(model, imgs_vis, writer, epoch, vis_dir, config):
    noise = torch.randn(imgs_vis.size(0), config.noise_dim).cuda()
    # noise = multi_modes_noise(imgs_vis.size(0), config.noise_dim)
    rand_label = torch.randn(imgs_vis.size(0), 1)
    imgs_vis_aug, rand_label = model.stn(noise, imgs_vis, rand_label)
    imgs_vis_aug = imgs_vis_aug.cpu()
    # print('imgs_vis_aug shape: {}'.format(imgs_vis_aug.size()))
    grid1 = make_grid(imgs_vis_aug[:imgs_vis.size(0)],
                     nrow=int(math.sqrt(imgs_vis.size(0))),
                     normalize=config.vis_nrm, padding=1, pad_value=1)
    grid2 = make_grid(imgs_vis_aug[imgs_vis.size(0):],
                      nrow=int(math.sqrt(imgs_vis.size(0))),
                      normalize=config.vis_nrm, padding=1, pad_value=1)
    # print('grid1 size: {}'.format(grid1.size()))
    # print('grid2 size: {}'.format(grid1.size()))
    grid = torch.cat([grid1, grid2], dim=2)
    # print('grid size: {}'.format(grid.size()))
    # exit()
    # print('imgs_vis_aug shape: {}'.format(grid.size()))
    # writer.add_image('imgs_aug_{}'.format(epoch), grid, 0)
    writer.add_image('imgs_stn', grid, epoch)
    save_image(grid, os.path.join(vis_dir,
                                  'stn_imgs_{}'.format(epoch) + '.png'))


# save visualized images for noise input stn
def save_vis_imgs_2(model, imgs_vis, writer, epoch, vis_dir, config):
    noise = torch.randn(imgs_vis.size(0), config.noise_dim).cuda()
    imgs_vis_aug = model.aug_net(noise, imgs_vis).cpu()
    # print('imgs_vis_aug shape: {}'.format(imgs_vis_aug.size()))
    grid = make_grid(imgs_vis_aug, nrow=int(math.sqrt(imgs_vis_aug.size(0))),
                     normalize=config.vis_nrm, padding=1, pad_value=1)
    # print('imgs_vis_aug shape: {}'.format(grid.size()))
    # writer.add_image('imgs_aug_{}'.format(epoch), grid, 0)
    writer.add_image('imgs_stn', grid, epoch)
    save_image(grid, os.path.join(vis_dir,
                                  'stn_imgs_{}'.format(epoch) + '.png'))


# save visualized images for noise input stn with double cycle loss
def save_vis_imgs_3(model, imgs_vis, writer, epoch, vis_dir, config):
    noise = torch.randn(imgs_vis.size(0), config.noise_dim).cuda()
    # noise = multi_modes_noise(imgs_vis.size(0), config.noise_dim)
    rand_label = torch.randn(imgs_vis.size(0), 1)
    imgs_vis_aug, rand_label = model.aug_net(noise, imgs_vis, rand_label)
    imgs_vis_aug = imgs_vis_aug.cpu()
    # print('imgs_vis_aug shape: {}'.format(imgs_vis_aug.size()))
    grid1 = make_grid(imgs_vis_aug[:imgs_vis.size(0)],
                     nrow=int(math.sqrt(imgs_vis.size(0))),
                     normalize=config.vis_nrm, padding=1, pad_value=1)
    grid2 = make_grid(imgs_vis_aug[imgs_vis.size(0):],
                      nrow=int(math.sqrt(imgs_vis.size(0))),
                      normalize=config.vis_nrm, padding=1, pad_value=1)
    # print('grid1 size: {}'.format(grid1.size()))
    # print('grid2 size: {}'.format(grid1.size()))
    grid = torch.cat([grid1, grid2], dim=2)
    # print('grid size: {}'.format(grid.size()))
    # exit()
    # print('imgs_vis_aug shape: {}'.format(grid.size()))
    # writer.add_image('imgs_aug_{}'.format(epoch), grid, 0)
    writer.add_image('imgs_stn', grid, epoch)
    save_image(grid, os.path.join(vis_dir,
                                  'stn_imgs_{}'.format(epoch) + '.png'))


# save visualized images for noise input stn with double cycle loss and multi augnets
# single noise used for all stns
def save_vis_imgs_4(model, imgs_vis, writer, epoch, vis_dir, config):
    noise = torch.randn(imgs_vis.size(0), config.noise_dim).cuda()
    rand_label = torch.randn(imgs_vis.size(0), 1)

    grid_list = []
    for k in range(len(model.aug_net_list)):
        imgs_vis_aug, rand_label = model.aug_net_list[k](noise, imgs_vis, rand_label)
        imgs_vis_aug = imgs_vis_aug.cpu()
        # print('imgs_vis_aug shape: {}'.format(imgs_vis_aug.size()))
        grid1 = make_grid(imgs_vis_aug[:imgs_vis.size(0)],
                         nrow=int(math.sqrt(imgs_vis.size(0))),
                         normalize=config.vis_nrm, padding=1, pad_value=1)
        grid2 = make_grid(imgs_vis_aug[imgs_vis.size(0):],
                          nrow=int(math.sqrt(imgs_vis.size(0))),
                          normalize=config.vis_nrm, padding=1, pad_value=1)
        # print('grid1 size: {}'.format(grid1.size()))
        # print('grid2 size: {}'.format(grid1.size()))
        grid = torch.cat([grid1, grid2], dim=2)
        grid_list.append(grid)

    grid = torch.cat(grid_list, dim=1)
    # print('grid size: {}'.format(grid.size()))
    # exit()
    # print('imgs_vis_aug shape: {}'.format(grid.size()))
    # writer.add_image('imgs_aug_{}'.format(epoch), grid, 0)
    writer.add_image('imgs_stn', grid, epoch)
    save_image(grid, os.path.join(vis_dir,
                                  'stn_imgs_{}'.format(epoch) + '.png'))


# save visualized images for noise input stn with double cycle loss and multi augnets
# each stn has an independent noise
def save_vis_imgs_5(model, imgs_vis, writer, epoch, vis_dir, config):

    grid_list = []
    for k in range(len(model.aug_net_list)):
        noise = torch.randn(imgs_vis.size(0), config.noise_dim_list[k]).cuda()
        rand_label = torch.randn(imgs_vis.size(0), 1)
        imgs_vis_aug, rand_label = model.aug_net_list[k](noise, imgs_vis, rand_label)
        imgs_vis_aug = imgs_vis_aug.cpu()
        # print('imgs_vis_aug shape: {}'.format(imgs_vis_aug.size()))
        grid1 = make_grid(imgs_vis_aug[:imgs_vis.size(0)],
                         nrow=int(math.sqrt(imgs_vis.size(0))),
                         normalize=config.vis_nrm, padding=1, pad_value=1)
        grid2 = make_grid(imgs_vis_aug[imgs_vis.size(0):],
                          nrow=int(math.sqrt(imgs_vis.size(0))),
                          normalize=config.vis_nrm, padding=1, pad_value=1)
        # print('grid1 size: {}'.format(grid1.size()))
        # print('grid2 size: {}'.format(grid1.size()))
        grid = torch.cat([grid1, grid2], dim=2)
        grid_list.append(grid)

    grid = torch.cat(grid_list, dim=1)
    # print('grid size: {}'.format(grid.size()))
    # exit()
    # print('imgs_vis_aug shape: {}'.format(grid.size()))
    # writer.add_image('imgs_aug_{}'.format(epoch), grid, 0)
    writer.add_image('imgs_stn', grid, epoch)
    save_image(grid, os.path.join(vis_dir,
                                  'stn_imgs_{}'.format(epoch) + '.png'))


# save visualized images for adv aug without any augmentation network.
def save_vis_imgs_6(model, imgs_vis, labels_vis, writer, epoch, vis_dir, normalize, bn_type=None):
    imgs_vis_aug = model.texture_aug(imgs_vis, labels_vis, bn_type).cpu()
    # print('imgs_vis_aug shape: {}'.format(imgs_vis_aug.size()))
    grid = make_grid(imgs_vis_aug, nrow=int(math.sqrt(imgs_vis_aug.size(0))),
                     normalize=normalize, padding=1, pad_value=1)
    # print('imgs_vis_aug shape: {}'.format(grid.size()))
    # writer.add_image('imgs_aug_{}'.format(epoch), grid, 0)
    writer.add_image('imgs_aug', grid, epoch)
    save_image(grid, os.path.join(vis_dir,
                                  'aug_imgs_{}'.format(epoch) + '.png'))


class RandomNoise(object):
    def __init__(self, min, max, probability=0.5):
        self.min = min
        self.max = max
        self.probability = probability
    def __call__(self, img):
        if np.random.random() <= self.probability:
            img = img + torch.randn_like(img)
            return torch.clamp(img, min=self.min, max=self.max)
        return img


def multi_modes_noise(batch_size, noise_dim, mode_num=5):
    noise = torch.randn(batch_size, noise_dim).cuda()
    if mode_num > 1:
        modes = torch.randint(mode_num, (batch_size, 1), dtype=torch.float).repeat(1, noise_dim).cuda()
        noise = noise + modes

    return noise


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.size()) == 1 or name.endswith(".bias") or name in skip_list:
            # print('param name: {}'.format(name))
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

