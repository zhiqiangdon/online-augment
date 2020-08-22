import torch
from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn

# from models.resnet_cifar import *
# from models.wide_resnet_old import *
# from models.resnext_cifar import *
# from models.densenet_cifar import *
# from models.mnist import *
# from models.resnet_cifar_multi_bn import *
# from models.wide_resnet_multi_bn import *

from models.resnet import ResNet
from models.pyramidnet import PyramidNet
from models.shakeshake.shake_resnet import ShakeResNet
from models.wideresnet import WideResNet
from models.shakeshake.shake_resnext import ShakeResNeXt
from models.wideresnet_multibn import WideResNetMultiBN
from models.shakeshake.shake_resnet_multibn import ShakeResNetMultiBN
from models.pyramidnet_multibn import PyramidNetMultiBN
from models.resnet_multibin import ResNetMultiBN

def get_model(config, num_class=10, bn_types=None, data_parallel=True):
    name = config.model
    print('model name: {}'.format(name))
    print('bn_types: {}'.format(bn_types))
    if name == 'resnet50':
        if bn_types is None:
            model = ResNet(dataset='imagenet', depth=50, num_classes=num_class, bottleneck=True)
        else:
            model = ResNetMultiBN(dataset='imagenet', depth=50, num_classes=num_class,
                                  bn_types=bn_types, bottleneck=True)
    elif name == 'resnet200':
        if bn_types is None:
            model = ResNet(dataset='imagenet', depth=200, num_classes=num_class, bottleneck=True)
        else:
            model = ResNetMultiBN(dataset='imagenet', depth=200, num_classes=num_class,
                                  bn_types=bn_types, bottleneck=True)
    elif name == 'wresnet40_2':
        if bn_types is None:
            model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class)
        else:
            raise Exception('unimplemented error')
    elif name == 'wresnet28_10':
        if bn_types is None:
            model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class)
        else:
            model = WideResNetMultiBN(28, 10, dropout_rate=0.0, num_classes=num_class, bn_types=bn_types)
    elif name == 'shakeshake26_2x32d':
        if bn_types is None:
            model = ShakeResNet(26, 32, num_class)
        else:
            model = ShakeResNetMultiBN(26, 32, num_class, bn_types)
    elif name == 'shakeshake26_2x64d':
        if bn_types is None:
            model = ShakeResNet(26, 64, num_class)
        else:
            model = ShakeResNetMultiBN(26, 64, num_class, bn_types)
    elif name == 'shakeshake26_2x96d':
        if bn_types is None:
            model = ShakeResNet(26, 96, num_class)
        else:
            model = ShakeResNetMultiBN(26, 96, num_class, bn_types)
    elif name == 'shakeshake26_2x112d':
        if bn_types is None:
            model = ShakeResNet(26, 112, num_class)
        else:
            model = ShakeResNetMultiBN(26, 112, num_class, bn_types)
    elif name == 'shakeshake26_2x96d_next':
        if bn_types is None:
            model = ShakeResNeXt(26, 96, 4, num_class)
        else:
            raise Exception('unimplemented error')

    elif name == 'pyramid':
        if bn_types is None:
            model = PyramidNet('cifar10', depth=config.pyramidnet_depth,
                               alpha=config.pyramidnet_alpha,
                               num_classes=num_class, bottleneck=True)
        else:
            model = PyramidNetMultiBN('cifar10', depth=config.pyramidnet_depth,
                               alpha=config.pyramidnet_alpha,
                               num_classes=num_class, bottleneck=True, bn_types=bn_types)
    else:
        raise NameError('no model named, %s' % name)

    if data_parallel:
        model = model.cuda()
        model = DataParallel(model)
    else:
        import horovod.torch as hvd
        device = torch.device('cuda', hvd.local_rank())
        model = model.to(device)
    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'reduced_cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]
