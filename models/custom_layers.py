import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class MultiBatchNorm(nn.Module):
    def __init__(self, dim, types, num_features, momentum=None):
        assert isinstance(types, list) and len(types) > 1
        assert 'base' in types
        assert dim in ('1d', '2d')
        super(MultiBatchNorm, self).__init__()
        self.types = types

        if dim == '1d':
            if momentum is not None:
                self.bns = nn.ModuleDict([[t, nn.BatchNorm1d(num_features, momentum=momentum)] for t in types])
            else:
                self.bns = nn.ModuleDict([[t, nn.BatchNorm1d(num_features)] for t in types])
        elif dim == '2d':
            if momentum is not None:
                self.bns = nn.ModuleDict([[t, nn.BatchNorm2d(num_features, momentum=momentum)] for t in types])
            else:
                self.bns = nn.ModuleDict([[t, nn.BatchNorm2d(num_features)] for t in types])

        self.t = 'base'

    def forward(self, x):
        # print('bn type: {}'.format(self.t))
        assert self.t in self.types
        out = self.bns[self.t](x)
        self.t = 'base'
        return out
