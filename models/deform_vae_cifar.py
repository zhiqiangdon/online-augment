
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import models

class Conv3x3_BN_RELU(nn.Sequential):
    def __init__(self, in_num, out_num, stride):
        super(Conv3x3_BN_RELU, self).__init__()
        self.add_module('conv', nn.Conv2d(in_num, out_num, kernel_size=3,
                                          stride=stride, padding=1, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

class Deconv3x3_BN_RELU(nn.Sequential):
    def __init__(self, in_num, out_num, stride):
        super(Deconv3x3_BN_RELU, self).__init__()
        if stride == 1:
            output_padding = 0
        elif stride == 2:
            output_padding = 1

        self.add_module('conv', nn.ConvTranspose2d(in_num, out_num, kernel_size=3,
                                                   stride=stride, padding=1,
                                                   output_padding=output_padding,
                                                   bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

class FC_BN_RELU(nn.Sequential):
    def __init__(self, in_num, out_num):
        super(FC_BN_RELU, self).__init__()
        self.add_module('fc', nn.Linear(in_num, out_num))
        self.add_module('bn', nn.BatchNorm1d(out_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('relu', nn.LeakyReLU(0.2, inplace=True))


class VAE(nn.Module):
    def __init__(self, z_dim=16, feat_dim=512):
        print('using conv vae ...')
        super(VAE, self).__init__()
        self.feat_dim = feat_dim
        self.z_dim = z_dim
        # self.smooth_weight = smooth_weight
        print('z_dim: {}'.format(z_dim))
        print('fea_dim: {}'.format(feat_dim))
        # print('smooth weight: {}'.format(self.smooth_weight))
        # assert self.smooth_weight >= 0
        # exit()
        self.encode_convs = nn.Sequential(
            Conv3x3_BN_RELU(3, 32, 1), # 32x32
            # nn.MaxPool2d(2, stride=2),
            Conv3x3_BN_RELU(32, 64, 2), # 16x16
            Conv3x3_BN_RELU(64, 64, 1),
            # nn.MaxPool2d(2, stride=2),
            Conv3x3_BN_RELU(64, 128, 2), # 8x8
            Conv3x3_BN_RELU(128, 128, 1),
            # nn.MaxPool2d(2, stride=2),
            # Conv3x3_BN_RELU(128, 256, 2), # 4x4
            # Conv3x3_BN_RELU(256, 256, 1),
            # # nn.MaxPool2d(2, stride=2),
            # Conv3x3_BN_RELU(256, 512, 2), # 2x2
            # Conv3x3_BN_RELU(512, 512, 1),
        )
        self.encode_fc1 = FC_BN_RELU(128 * 8 * 8, self.feat_dim)

        self.encode_fc_z_mu = nn.Linear(self.feat_dim, self.z_dim)
        self.encode_fc_z_var = nn.Linear(self.feat_dim, self.z_dim)

        self.decode_fc1 = FC_BN_RELU(self.z_dim, self.feat_dim)
        self.decode_fc2 = nn.Linear(self.feat_dim, 128 * 8 * 8)
        # self.sigmoid = nn.Sigmoid()

        self.decode_convs = nn.Sequential(
            # Conv3x3_BN_RELU(256, 512, 2), # 2x2
            # Conv3x3_BN_RELU(512, 512, 1),

            # Conv3x3_BN_RELU(128, 256, 2), # 4x4
            # Conv3x3_BN_RELU(256, 256, 1),

            Deconv3x3_BN_RELU(128, 128, 1),  # 8x8
            Deconv3x3_BN_RELU(128, 64, 2),

            Deconv3x3_BN_RELU(64, 64, 1),  # 16x16
            Deconv3x3_BN_RELU(64, 32, 2),

            Deconv3x3_BN_RELU(32, 32, 1),
            # Deconv3x3_BN_RELU(32, 3, 1),  # 32x32
        )
        # self.last_layer = nn.Conv2d(3, 3, kernel_size=3, stride=1,
        #                             padding=1, bias=False)
        self.last_layer = nn.Conv2d(32, 2, kernel_size=1, stride=1, bias=True)

        self.mse_loss = nn.MSELoss()

        self.id_theta = torch.tensor([[[1, 0, 0],
                                       [0, 1, 0]]], dtype=torch.float32).cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # conv_count += 1
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # bn_count += 1
        # initialize the last layer with zeros
        self.last_layer.weight.data.zero_()
        self.last_layer.bias.data.zero_()
        # TODO: whether need some specific initialization for predicting theta?
        # self.encode_fc_theta_mu[-1].weight.data.zero_()
        # self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def encode(self, x):
        x_conv = self.encode_convs(x)
        # print('x_conv size: {}'.format(x_conv.size()))
        x_fc = self.encode_fc1(x_conv.view(x.size(0), -1))
        return self.encode_fc_z_mu(x_fc), self.encode_fc_z_var(x_fc)

    def decode(self, z):
        z_fc = self.decode_fc1(z)
        z_fc = self.decode_fc2(z_fc).view(-1, 128, 8, 8)
        # print('z_fc size: {}'.format(z_fc.size()))
        x = self.decode_convs(z_fc)
        x = self.last_layer(x)
        # x = torch.sigmoid(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def loss(self, grid_delta, z_mu, z_logvar):
        MSE = torch.mean(grid_delta ** 2)
        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_z = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp()) / grid_delta.size(0)
        return MSE + KLD_z

    def smooth_loss(self, grid_delta, norm='l2'):
        dx = torch.abs(grid_delta[:, :, 1:, :] - grid_delta[:, :, :-1, :])
        dy = torch.abs(grid_delta[:, 1:, :, :] - grid_delta[:, :-1, :, :])
        if norm == 'l2':
            loss = torch.mean(dx ** 2) + torch.mean(dy ** 2)
        elif norm == 'l1':
            loss = torch.mean(dx) + torch.mean(dy)
        else:
            raise Exception('invalid norm: {}'.format(norm))

        return loss/2.0

    def forward(self, x, require_loss=False, require_grid=False):
        # print('x size: {}'.format(x.size()))
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        grid_delta = self.decode(z)

        grid_id = F.affine_grid(self.id_theta.repeat(x.size(0), 1, 1), x.size())
        grid_delta = grid_delta.permute(0, 2, 3, 1)
        grid = grid_id + grid_delta
        x_deform = F.grid_sample(x, grid, mode='bilinear', padding_mode='border')
        # use this when the groundtruth also requires the corresponding deformation
        # label_deform = F.grid_sample(label.float(), grid, mode='nearest', padding_mode='border').long()

        if not require_loss:
            if require_grid:
                return x_deform, grid
            else:
                return x_deform
        else:
            return x_deform, self.loss(grid_delta, z_mu, z_logvar), self.smooth_loss(grid_delta)