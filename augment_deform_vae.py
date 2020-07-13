import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import utils

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Augment(object):

  def __init__(self, target_net, aug_net, config):

    # loss function
    self.criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    # print('target net lr: {}'.format(config.lr))
    # print('target optimizer momentum: {}'.format(config.momentum))
    # print('target optimizer weight decay: {}'.format(config.weight_decay))
    # if config.decay_type is None:
    #   params = target_net.parameters()
    # elif config.decay_type == 'no_bn':
    #   params = utils.add_weight_decay(target_net, config.weight_decay)
    # else:
    #   raise Exception('unknown decay type: {}'.format(config.decay_type))
    self.target_net_optim = optim.SGD(target_net.parameters(), config.lr,
                                      momentum=config.momentum,
                                      weight_decay=config.weight_decay,
                                      nesterov=True)
    for group in self.target_net_optim.param_groups:
      print('target net lr: {}, weight_decay: {}, momentum: {}, nesterov: {}'
            .format(group['lr'], group['weight_decay'], group['momentum'], group['nesterov']))

    print('training epochs: {}'.format(config.epochs))
    # lr scheduler
    if config.lr_scheduler == 'cosine':
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.target_net_optim,
                                                                       T_max=float(config.epochs),
                                                                       eta_min=0.)
    else:
        raise ValueError('invalid lr_schduler: {}'.format(config.lr_scheduler))

    # self.network_momentum = args.momentum
    self.target_net = target_net
    self.aug_net = aug_net
    # self.criterion = criterion
    self.args = config
    print('adv weight for deformation: {}'.format(self.args.adv_weight_deform))
    print('reconstruction weight for deformation: {}'.format(self.args.div_weight_deform))
    print('smooth weight: {}'.format(self.args.smooth_weight))
    assert self.args.adv_weight_deform >= 0
    assert self.args.div_weight_deform >= 0
    assert self.args.smooth_weight >= 0
    print('aug_net_lr: {}'.format(config.aug_net_lr))
    print('aug net adam optimizer beta1: {}'.format(config.adam_beta1))
    print('aug_net_weight_decay: {}'.format(config.aug_net_weight_decay))
    self.aug_net_optim = torch.optim.Adam(aug_net.parameters(),
                                          lr=config.aug_net_lr, betas=(config.adam_beta1, 0.999),
                                          weight_decay=config.aug_net_weight_decay)

  # def call_aug_net(self, data, label, require_loss=False):
  #   if require_loss:
  #     if 'vae' in self.args.exp_type and self.args.vae_type == 'condition':
  #       # print('using conditional vae ...')
  #       # exit()
  #       data_aug, loss = self.aug_net(data, F.one_hot(label, 10).type(torch.cuda.FloatTensor), require_loss=True)
  #     else:
  #       data_aug, loss = self.aug_net(data, require_loss=True)
  #
  #     return data_aug, loss
  #   else:
  #     if 'vae' in self.args.exp_type and self.args.vae_type == 'condition':
  #       data_aug = self.aug_net(data, F.one_hot(label, 10).type(torch.cuda.FloatTensor))
  #     else:
  #       data_aug = self.aug_net(data)
  #
  #     return data_aug

  def _compute_unrolled_model(self, loss, eta):

    theta = _concat(self.target_net.parameters()).data
    dtheta = _concat(torch.autograd.grad(loss, self.target_net.parameters(), retain_graph=True)).data
    # dtheta = _concat([v.grad for v in self.target_net.parameters()]).data
    if self.args.weight_decay != 0:
      dtheta.add_(self.args.weight_decay, theta)
    if self.args.momentum != 0:
      try:
        moment = _concat(self.target_net_optim.state[v]['momentum_buffer'] for v in self.target_net.parameters()).mul_(self.args.momentum)
      except:
        # setting zeros is consistent with the original momentum optimizer
        moment = torch.zeros_like(theta)
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
    return unrolled_model

  # def _compute_unrolled_model(self, eta):
  #
  #
  #   theta = _concat([v.data for v in self.target_net.parameters()])
  #   try:
  #     moment = _concat(self.target_net_optim.state[v]['momentum_buffer'] for v in self.target_net.parameters()).mul_(self.args.momentum)
  #   except:
  #     moment = torch.zeros_like(theta)
  #   dtheta = _concat([v.grad.data for v in self.target_net.parameters()]) + self.args.weight_decay * theta
  #
  #   unrolled_target_net = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
  #
  #   return unrolled_target_net

  def step(self, input_train, target_train, input_valid, target_valid, unrolled):
    self.aug_net_optim.zero_grad()
    if unrolled:
        # self._backward_step_unrolled(input_train, target_train, input_valid, target_valid)
        loss_adv, loss_div = self._backward_step_unrolled(input_train, target_train, input_valid, target_valid)
    else:
        self._backward_step(input_valid, target_valid)
    self.aug_net_optim.step()
    return loss_adv, loss_div

  def _backward_step(self, input_valid, target_valid):
    output_valid = self.target_net(input_valid)
    loss = self.criterion(output_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid):
    # input_train_aug, div_loss = self.aug_net(input_train, require_loss=True)
    input_train_aug, div_loss = self.aug_net(input_train, require_loss=True)
    output_train = self.target_net(input_train_aug)
    loss_train = self.criterion(output_train, target_train)
    # loss_train_2 = loss_train + div_loss * self.args.div_weight
    # print('div_loss: {}'.format(div_loss))
    # print('loss_train: {}'.format(loss_train))
    # print('loss_train_2: {}'.format(loss_train_2))
    # print('self.args.div_weight: {}'.format(self.args.div_weight))
    # exit()
    # loss_train_2.backward()
    eta = self.target_net_optim.param_groups[0]['lr']
    # eta = self.lr_scheduler.get_lr()[0]
    unrolled_target_net = self._compute_unrolled_model(loss_train, eta)

    # # check whether the unrolled_target_net is different from the original target_net
    # output2 = unrolled_target_net(input_train)
    # loss2 = self.criterion(output2, target_train)
    # print('loss1: {:4f}'.format(loss_train))
    # print('loss2: {:4f}'.format(loss2))
    # exit()
    output_valid = unrolled_target_net(input_valid)
    unrolled_loss = self.criterion(output_valid, target_valid)

    unrolled_loss.backward()
    #
    loss_train_aug = -loss_train * self.args.adv_weight + div_loss * self.args.div_weight
    dalpha = torch.autograd.grad(loss_train_aug, self.aug_net.parameters())
    # dalpha = [per_grad.data.clamp_(min=-1, max=1) for per_grad in dalpha]

    vector = [v.grad.data for v in unrolled_target_net.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train, r=self.args.val_r)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.aug_net.parameters(), dalpha):
      if v.grad is None:
        # print('grad is none. existing ...')
        # exit()
        v.grad = g.detach()
      else:
        v.grad.data.copy_(g.data)
    return -loss_train * self.args.adv_weight, div_loss * self.args.div_weight

  def _construct_model_from_theta(self, theta):
    # print('type of theta: {}'.format(type(theta)))
    theta = nn.Parameter(theta)
    target_net_new = utils.build_model(self.args)
    # .state_dict() stores all the persistent buffers (e.g. running averages), which are not included in .parameters()
    model_dict = self.target_net.state_dict()

    params, offset = {}, 0
    for k, v in self.target_net.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      # print('type of params[k]: {}'.format(type(params[k])))
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    target_net_new.load_state_dict(model_dict)
    return target_net_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=2e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.target_net.parameters(), vector):
      p.data.add_(R, v)

    # input_aug = self.aug_net(input)
    input_aug = self.call_aug_net(input, target)
    output_aug = self.target_net(input_aug)
    loss = self.criterion(output_aug, target)
    grads_p = torch.autograd.grad(loss, self.aug_net.parameters())

    for p, v in zip(self.target_net.parameters(), vector):
      p.data.sub_(2*R, v)

    # input_aug = self.aug_net(input)
    input_aug = self.call_aug_net(input, target)
    output_aug = self.target_net(input_aug)
    loss = self.criterion(output_aug, target)
    grads_n = torch.autograd.grad(loss, self.aug_net.parameters())

    # recover the original weights in self.target_net
    for p, v in zip(self.target_net.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]