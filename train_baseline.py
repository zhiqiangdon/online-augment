import os
import time
import shutil
from collections import defaultdict
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn

from utils import AverageMeter
import utils
# import cifar_models as cifar_models
# from torch.utils.data.sampler import SubsetRandomSampler
# import json
# from torchvision.utils import make_grid, save_image
# import math
# from warmup_scheduler import GradualWarmupScheduler
from data import get_dataloaders
from models import get_model, num_class

def train_and_validate(config):

    # data loaders
    trainloader, testloader = get_dataloaders(config)

    # model
    model = get_model(config, num_class(config.dataset))

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    # if config.decay_type is None:
    #     params = model.parameters()
    # elif config.decay_type == 'no_bn':
    #     params = utils.add_weight_decay(model, config.weight_decay)
    # else:
    #     raise Exception('unknown decay type: {}'.format(config.decay_type))
    optimizer = optim.SGD(model.parameters(), config.lr,
                          momentum=config.momentum,
                          weight_decay=config.weight_decay,
                          nesterov=True)
    # lr scheduler
    if config.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=float(config.epochs),
                                                                  eta_min=0.)
    else:
        raise ValueError('invalid lr_schduler: {}'.format(config.lr_scheduler))

    # if config.warmup_epoch > 0:
    #     print('using lr warmup scheduler...')
    #     lr_scheduler = GradualWarmupScheduler(
    #         optimizer,
    #         multiplier=config.warmup_multiplier,
    #         total_epoch=config.warmup_epoch,
    #         after_scheduler=lr_scheduler
    #     )

    start_epoch = 0
    best_test_acc = 0.0
    test_acc = 0.0
    if config.resume:
        best_test_acc, test_acc, start_epoch = \
            utils.load_checkpoint(config, model, optimizer)

    print('trainloader length: {}'.format(len(trainloader)))
    print('testloader length: {}'.format(len(testloader)))

    exp_dir = utils.get_log_dir_path(config.exp_dir, config.exp_id)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    print('exp_dir: {}'.format(exp_dir))
    log_file = os.path.join(exp_dir, 'log.txt')
    names = ['epoch', 'lr', 'Train Acc', 'Test Acc', 'Best Test Acc']
    with open(log_file, 'a') as f:
        f.write('batch size: {}\n'.format(config.batch_size))
        f.write('lr: {}\n'.format(config.lr))
        f.write('momentum: {}\n'.format(config.momentum))
        f.write('weight_decay: {}\n'.format(config.weight_decay))
        for per_name in names:
            f.write(per_name + '\t')
        f.write('\n')
    # print('=> Training the base model')
    # print('start_epoch {}'.format(start_epoch))
    # print(type(start_epoch))
    # exit()
    for epoch in range(start_epoch, config.epochs):
        # lr = adjust_learning_rate(optimizer, epoch, model.module, config)
        lr = optimizer.param_groups[0]['lr']
        print('lr: {}'.format(lr))
        # inner_lr = get_lr_cosine_decay(config, epoch)
        # print('inner_lr: {}'.format(inner_lr))
        # train for one epoch
        # print('training epoch ...')
        train_acc = train_epoch(trainloader, model, criterion, optimizer, lr_scheduler, epoch, config)
        # evaluate on test set
        # print('testing epoch ...')
        test_acc = validate_epoch(testloader, model, criterion, config)
        # remember best acc, evaluate on test set and save checkpoint
        is_best = test_acc > best_test_acc
        if is_best:
            best_test_acc = test_acc

        utils.save_checkpoint(model,{
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'test_acc': test_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, exp_dir)

        values = [train_acc, test_acc, best_test_acc]
        with open(log_file, 'a') as f:
            f.write('{:d}\t'.format(epoch))
            f.write('{:g}\t'.format(lr))
            for per_value in values:
                f.write('{:2.2f}\t'.format(per_value))
            f.write('\n')
        print('exp_dir: {}'.format(exp_dir))


def train_epoch(trainloader, model, criterion, optimizer, lr_scheduler, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    loader_len = len(trainloader)
    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        # grid = make_grid(input, nrow=int(math.sqrt(input.size(0))), normalize=False, padding=1, pad_value=1)
        # # print('imgs_vis_aug shape: {}'.format(grid.size()))
        # save_image(grid, os.path.join('gaussian_noise_imgs.png'))
        # exit()
        input, target = input.cuda(), target.cuda()

        # # debug, check if learning anything
        # print(list(model.module.fc.parameters())[0][0, 0].item())

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc = utils.accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(acc.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip and config.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        lr_scheduler.step(epoch + float(i+1) / loader_len)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            # print(target[:10])
            # exit()

    print(' * Acc {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


def validate_epoch(val_loader, model, criterion, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = utils.accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Acc {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg
