import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from .custom_layers import MultiBatchNorm

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, bn_types=None):
        super(WideBasic, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = MultiBatchNorm('2d', bn_types, in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = MultiBatchNorm('2d', bn_types, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNetMultiBNV2(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, bn_types):
        super(WideResNetMultiBNV2, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1, bn_types=bn_types)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2, bn_types=bn_types)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2, bn_types=bn_types)
        # self.bn1 = nn.BatchNorm2d(nStages[3])
        self.bn1 = MultiBatchNorm('2d', bn_types, nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)

        # self.apply(conv_init)

        conv_count = 0
        bn_count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_count += 1
            elif isinstance(m, nn.BatchNorm2d):
                bn_count += 1

        print('conv num: {}'.format(conv_count))
        print('bn num: {}'.format(bn_count))


    def _set_bn_type(self, t):
        count = 0
        for m in self.modules():
            if isinstance(m, MultiBatchNorm):
                m.t = t
                count += 1

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, bn_types):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, bn_types))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, t=None):
        if t is not None:
            self._set_bn_type(t)

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out