import torch
import torch.nn as nn

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, config):
        super(NormalizeByChannelMeanStd, self).__init__()
        mean, std = get_mean_std(config)
        # if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean).cuda()
        # if not isinstance(std, torch.Tensor):
        std = torch.tensor(std).cuda()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor, bn_type=None):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def get_mean_std(config):
    if config.dataset == 'mnist':
        mean = [0.1307]
        std = [0.3081]
    elif config.dataset == 'svhn':
        mean = [x / 255.0 for x in [109.9, 109.7, 113.8]]
        std = [x / 255.0 for x in [50.1, 50.6, 50.8]]
    elif 'cifar' in config.dataset:
        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    else:
        raise Exception('Error: unsopported dataset: {}'.format(config.dataset))

    return mean, std

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)