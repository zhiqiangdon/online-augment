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
from augment_comb import Augment

def train_and_validate(config):

    # data loaders
    trainloader, testloader = get_dataloaders(config)

    # model
    bn_types = ['base']
    if config.perturb_vae: bn_types.append('texture')
    if config.aug_stn: bn_types.append('stn')
    if config.deform_vae: bn_types.append('deform')

    # if config.bn_num == 1:
    #     target_net = get_model(config, num_class(config.dataset))
    # else:
    target_net = get_model(config, num_class(config.dataset), bn_types=bn_types)

    model = Augment(target_net=target_net, config=config)


    start_epoch = 0
    best_test_acc = 0.0
    test_acc = 0.0
    if config.resume:
        best_test_acc, test_acc, start_epoch = \
            utils.load_checkpoint(config, model.target_net, model.target_net_optim)

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
    print('target net grad clip: {}'.format(config.grad_clip))
    for epoch in range(start_epoch, config.epochs):
        # lr = adjust_learning_rate(optimizer, epoch, model.module, config)
        lr = model.target_net_optim.param_groups[0]['lr']
        print('lr: {}'.format(lr))
        # inner_lr = get_lr_cosine_decay(config, epoch)
        # print('inner_lr: {}'.format(inner_lr))
        # train for one epoch
        train_acc = train_epoch_multi_bns(trainloader, model, epoch, config)
        # evaluate on test set
        # print('testing epoch ...')
        test_acc = validate_epoch(testloader, model, config)
        # remember best acc, evaluate on test set and save checkpoint
        is_best = test_acc > best_test_acc
        if is_best:
            best_test_acc = test_acc

        utils.save_checkpoint(model,{
            'epoch': epoch + 1,
            'state_dict': model.target_net.state_dict(),
            'optimizer': model.target_net_optim.state_dict(),
            'perturb_vae_state_dict': model.perturb_vae.state_dict() if model.perturb_vae else None,
            'perturb_vae_optimizer': model.perturb_vae_optim.state_dict() if model.perturb_vae else None,
            'aug_stn_state_dict': model.aug_stn.state_dict() if model.aug_stn else None,
            'aug_stn_optimizer': model.aug_stn_optim.state_dict() if model.aug_stn else None,
            'deform_vae_state_dict': model.deform_vae.state_dict() if model.deform_vae else None,
            'deform_vae_optimizer': model.deform_vae_optim.state_dict() if model.deform_vae else None,
            'test_acc': test_acc,
            'best_test_acc': best_test_acc,
        }, is_best, exp_dir)

        values = [train_acc, test_acc, best_test_acc]
        with open(log_file, 'a') as f:
            f.write('{:d}\t'.format(epoch))
            f.write('{:g}\t'.format(lr))
            for per_value in values:
                f.write('{:2.2f}\t'.format(per_value))
            f.write('\n')
        print('exp_dir: {}'.format(exp_dir))


# not using implicit gradients from validation data
def train_epoch_multi_bns(trainloader, model, epoch, config):
    print('using function train_epoch_multi_bns...')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_adv = AverageMeter()
    losses_div = AverageMeter()
    # losses3 = AverageMeter()
    # losses4 = AverageMeter()
    top1 = AverageMeter()
    top1_aug_texture = AverageMeter()
    top1_aug_stn = AverageMeter()
    top1_aug_deform = AverageMeter()

    model.target_net.train()
    model.train_mode()
    loader_len = len(trainloader)
    end = time.time()
    for i, (input_list, target) in enumerate(trainloader):
        # measure data loading time
        # print('iter: {}'.format(i))
        data_time.update(time.time() - end)
        assert isinstance(input_list, list)
        assert len(input_list) == 2
        input, input_preaug = input_list[0], input_list[1]
        # print('input size: {}'.format(input.size()))
        # print('input_autoaug size: {}'.format(input_autoaug.size()))
        # print('target size: {}'.format(target.size()))
        input, input_preaug, target = input.cuda(), input_preaug.cuda(), target.cuda()

        # update aug_net and target_net
        model.target_net_optim.zero_grad()

        if model.perturb_vae:
            # print('updating perturb_vae...')
            zero_grad_no_bn(model)
            model.texture_step(input, target)
            # texture_grad = [p.grad.data.div_(-model.args.adv_weight_vae)
            #                 for n, p in model.target_net.named_parameters()
            #                 if p.grad is not None and 'bn' not in n]
            # texture_grad_bn = [p.grad.data.div_(-model.args.adv_weight_vae)
            #                    for n, p in model.target_net.named_parameters() if
            #                    p.grad is not None and 'bn' in n and 'texture' in n]

            texture_grad = []
            for n, p in model.target_net.named_parameters():
                if p.grad is not None:
                    if 'bn' not in n:
                        texture_grad.append(p.grad.data.div_(-model.args.adv_weight_vae))
                        # print('{}: {}'.format(n, texture_grad[-1].view(-1)[:5]))
                    elif 'texture' in n:
                        p.grad.data.div_(-model.args.adv_weight_vae)

        else:
            texture_grad = [torch.zeros_like(p.grad) for n, p in model.target_net.named_parameters()
                            if p.grad is not None and 'bn' not in n]

        if model.aug_stn:
            # print('updating stn...')
            zero_grad_no_bn(model)

            # # test
            # for grad in texture_grad:
            #     print('saved grad: {}'.format(grad.view(-1)[:5]))
            # exit()
            model.stn_step(input, target)
            # stn_grad = [p.grad.data.div_(-model.args.adv_weight_stn)
            #             for n, p in model.target_net.named_parameters()
            #             if p.grad is not None and 'bn' not in n]
            # stn_grad_bn = [p.grad.data.div_(-model.args.adv_weight_stn)
            #                for n, p in model.target_net.named_parameters() if
            #                p.grad is not None and 'bn' in n and 'stn' in n]
            stn_grad = []
            for n, p in model.target_net.named_parameters():
                if p.grad is not None:
                    if 'bn' not in n:
                        stn_grad.append(p.grad.data.div_(-model.args.adv_weight_stn))
                    elif 'stn' in n:
                        p.grad.data.div_(-model.args.adv_weight_stn)
        else:
            stn_grad = [torch.zeros_like(p.grad) for n, p in model.target_net.named_parameters()
                        if p.grad is not None and 'bn' not in n]

        if model.deform_vae:
            # print('updating deform_vae...')
            zero_grad_no_bn(model)
            model.deform_step(input, target)
            # deform_grad = [p.grad.data.div_(-model.args.adv_weight_deform)
            #                for n, p in model.target_net.named_parameters()
            #                if p.grad is not None and 'bn' not in n]
            # deform_grad_bn = [p.grad.data.div_(-model.args.adv_weight_deform)
            #                   for n, p in model.target_net.named_parameters() if
            #                   p.grad is not None and 'bn' in n and 'deform' in n]

            deform_grad = []
            for n, p in model.target_net.named_parameters():
                if p.grad is not None:
                    if 'bn' not in n:
                        deform_grad.append(p.grad.data.div_(-model.args.adv_weight_deform))
                    elif 'deform' in n:
                        p.grad.data.div_(-model.args.adv_weight_deform)
        else:
            deform_grad = [torch.zeros_like(p.grad) for n, p in model.target_net.named_parameters()
                           if p.grad is not None and 'bn' not in n]

        # # update target net
        # model.target_net_optim.zero_grad()
        # loss_aug_texture = 0
        # if model.texture_vae:
        #     # print('using texture_vae...')
        #     input_aug_texture = model.texture_vae(input)
        #     output_aug_texture = model.target_net(input_aug_texture.detach(), 'texture')
        #     loss_aug_texture = model.criterion(output_aug_texture, target)
        #     acc_aug_texture = utils.accuracy(output_aug_texture, target)[0]
        #     top1_aug_texture.update(acc_aug_texture.item(), input.size(0))
        #     # print('texture_vae aug loss: {:.4f}'.format(loss_aug_texture.item()))
        #
        # loss_aug_stn = 0
        # if model.stn:
        #     # print('using stn...')
        #     input_aug_stn, target_aug_stn = model.stn(noise, input, target)
        #     output_aug_stn = model.target_net(input_aug_stn.detach(), 'stn')
        #     loss_aug_stn = model.criterion(output_aug_stn, target_aug_stn.detach())
        #     acc_aug_stn = utils.accuracy(output_aug_stn, target_aug_stn)[0]
        #     top1_aug_stn.update(acc_aug_stn.item(), input.size(0))
        #     # print('stn aug loss: {:.4f}'.format(loss_aug_stn.item()))
        #
        # loss_aug_deform = 0
        # if model.deform_vae:
        #     # print('using deform_vae...')
        #     input_aug_deform = model.deform_vae(input)
        #     output_aug_deform = model.target_net(input_aug_deform.detach(), 'deform')
        #     loss_aug_deform = model.criterion(output_aug_deform, target)
        #     acc_aug_deform = utils.accuracy(output_aug_deform, target)[0]
        #     top1_aug_deform.update(acc_aug_deform.item(), input.size(0))
        #     # print('deform aug loss: {:.4f}'.format(loss_aug_deform.item()))

        # model.target_net_optim.zero_grad()
        target_params = [p for n, p in model.target_net.named_parameters()
                         if p.grad is not None and 'bn' not in n]
        # target_params_texture_bn = [p for n, p in model.target_net.named_parameters()
        #                             if p.grad is not None and 'bn' in n and 'texture' in n]
        # target_params_stn_bn = [p for n, p in model.target_net.named_parameters()
        #                         if p.grad is not None and 'bn' in n and 'stn' in n]
        # target_params_deform_bn = [p for n, p in model.target_net.named_parameters()
        #                            if p.grad is not None and 'bn' in n and 'deform' in n]

        count = 0
        for p, t_g, s_g, d_g in zip(target_params, texture_grad, stn_grad, deform_grad):
            # print('iter: {}, p.grad size: {}, t_g size: {}, s_g size: {}, d_g size: {}'
            #       .format(count, p.grad.data.size(), t_g.data.size(), s_g.data.size(), d_g.data.size()))
            print('texture_grad: {}'.format(t_g.data.view(-1)[:5]))
            print('stn_grad: {}'.format(s_g.data.view(-1)[:5]))
            print('texture_grad and stn_grad diff: {}'.format(torch.dist(t_g.data, s_g.data)))
            # print('texture_grad and deform_grad diff: {}'.format(torch.dist(t_g.data, d_g.data)))

            p.grad.data.copy_(t_g.data + s_g.data + d_g.data)
            count += 1
        exit()
        # for p, t_g in zip(target_params_texture_bn, texture_grad_bn):
        #     # print('iter: {}, p.grad size: {}, t_g size: {}, s_g size: {}, d_g size: {}'
        #     #       .format(count, p.grad.data.size(), t_g.data.size(), s_g.data.size(), d_g.data.size()))
        #     p.grad.data.copy_(t_g.data)
        #     count += 1
        #
        # for p, s_g in zip(target_params_stn_bn, stn_grad_bn):
        #     # print('iter: {}, p.grad size: {}, t_g size: {}, s_g size: {}, d_g size: {}'
        #     #       .format(count, p.grad.data.size(), t_g.data.size(), s_g.data.size(), d_g.data.size()))
        #     p.grad.data.copy_(s_g.data)
        #     count += 1
        #
        # for p, d_g in zip(target_params_deform_bn, deform_grad_bn):
        #     # print('iter: {}, p.grad size: {}, t_g size: {}, s_g size: {}, d_g size: {}'
        #     #       .format(count, p.grad.data.size(), t_g.data.size(), s_g.data.size(), d_g.data.size()))
        #     p.grad.data.copy_(d_g.data)
        #     count += 1


        output_preaug = model.target_net(input_preaug, 'base')
        loss_preaug = model.criterion(output_preaug, target)
        # print('pre aug loss: {:.4f}'.format(loss_preaug.item()))
        # exit()
        # loss = loss_aug_texture + loss_aug_stn + loss_aug_deform + loss_preaug
        loss_preaug.backward()

        if config.grad_clip and config.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.target_net.parameters(), config.grad_clip)

        model.target_net_optim.step()

        # update lr
        model.lr_scheduler.step(epoch + float(i + 1) / loader_len)

        acc = utils.accuracy(output_preaug, target)[0]
        losses.update(loss_preaug.item(), input.size(0))
        top1.update(acc.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print('momentum_buffer: {}'.format(momentum_buffer[0][0, 0, 0:10]))

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Acc {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})\t'.format(
                   epoch, i, len(trainloader), top1=top1, losses=losses))
            # exit()

    print(' * Acc {top1.avg:.3f}% '.format(top1=top1))
    # exit()
    return top1.avg

def zero_grad_no_bn(model):
    for n, p in model.target_net.named_parameters():
        if p.grad is not None and 'bn' not in n:
            p.grad.detach_()
            p.grad.zero_()

def validate_epoch(val_loader, model, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.target_net.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model.target_net(input)
            loss = model.criterion(output, target)

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
