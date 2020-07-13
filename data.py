import logging
import os

import torch
import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from archive_policies import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, \
    fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet, autoaug_paper_svhn
from operations import *
# from common import get_logger
# from imagenet import ImageNet

# logger = get_logger('Augmentation')
# logger.setLevel(logging.INFO)
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def get_dataloaders(config, split=0, split_idx=0):
    if 'cifar' in config.dataset or 'svhn' in config.dataset:
        print('adding basic augmentation for {}'.format(config.dataset))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    elif 'imagenet' in config.dataset:
        print('adding basic augmentation for {}'.format(config.dataset))
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('unsupported dataset: {}'.format(config.dataset))

    # logger.debug('augmentation: {}'.format(config.aug_type))
    if config.aug_type == 'fa_reduced_cifar10':
        print('using fast autoaug policies for cifar10')
        transform_train.transforms.insert(0, Augmentation(fa_reduced_cifar10()))

    elif config.aug_type == 'fa_reduced_imagenet':
        print('using fast autoaug policies for imagenet')
        transform_train.transforms.insert(0, Augmentation(fa_resnet50_rimagenet()))

    elif config.aug_type == 'fa_reduced_svhn':
        print('using fast autoaug policies for svhn')
        transform_train.transforms.insert(0, Augmentation(fa_reduced_svhn()))

    elif config.aug_type == 'arsaug':
        transform_train.transforms.insert(0, Augmentation(arsaug_policy()))
    elif config.aug_type == 'autoaug_cifar10':
        print('using autoaug policies for cifar10')
        transform_train.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
    elif config.aug_type == 'autoaug_extend':
        print('using extended autoaug policies for cifar10')
        transform_train.transforms.insert(0, Augmentation(autoaug_policy()))
    elif config.aug_type == 'autoaug_svhn':
        print('using autoaug policies for svhn')
        transform_train.transforms.insert(0, Augmentation(autoaug_paper_svhn()))
    elif config.aug_type in ['default', 'inception', 'inception320']:
        raise Exception('unimplememted error')
    elif config.aug_type == 'basic':
        print('basic augmentation only')
        pass
    else:
        raise ValueError('not found augmentations: {}'.format(config.aug_type))

    if config.cutout > 0 and config.dataset != 'reduced_svhn':
        print('adding cutout augmentation')
        print('cutout size: {}'.format(config.cutout))
        transform_train.transforms.append(CutoutDefault(config.cutout))

    if config.dataset == 'cifar10':
        if config.exp_type == 'baseline':
            # here we check the exp_type rather than the bn_num because even when bn_num=1, we may still
            # use texture, stn, or deform aug along with the basic/autoaug. The input of our texture or deform
            # vae need the original data without augmentation as input.
            total_trainset = torchvision.datasets.CIFAR10(root=config.data_dir, train=True,
                                                          download=True, transform=transform_train)
        else:
            total_trainset = CustomTrainCifar10(root=config.data_dir, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(root=config.data_dir, train=False,
                                               download=True, transform=transform_test)
    elif config.dataset == 'reduced_cifar10':
        if config.exp_type == 'baseline':
            total_trainset = torchvision.datasets.CIFAR10(root=config.data_dir, train=True,
                                                          download=True, transform=transform_train)
        else:
            total_trainset = CustomTrainCifar10(root=config.data_dir, transform=transform_train)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=46000, random_state=0)   # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        testset = torchvision.datasets.CIFAR10(root=config.data_dir, train=False,
                                               download=True, transform=transform_test)
    elif config.dataset == 'cifar100':
        if config.exp_type == 'baseline':
            total_trainset = torchvision.datasets.CIFAR100(root=config.data_dir, train=True,
                                                           download=True, transform=transform_train)
        else:
            total_trainset = CustomTrainCifar100(root=config.data_dir, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(root=config.data_dir, train=False,
                                                download=True, transform=transform_test)
    elif config.dataset == 'reduced_cifar100':
        if config.exp_type == 'baseline':
            total_trainset = torchvision.datasets.CIFAR100(root=config.data_dir, train=True,
                                                           download=True, transform=transform_train)
        else:
            total_trainset = CustomTrainCifar100(root=config.data_dir, transform=transform_train)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=46000, random_state=0)  # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        testset = torchvision.datasets.CIFAR100(root=config.data_dir, train=False,
                                                download=True, transform=transform_test)
    elif config.dataset == 'svhn':
        if config.exp_type == 'baseline':
            trainset = torchvision.datasets.SVHN(root=config.data_dir, split='train',
                                                 download=True, transform=transform_train)
            extraset = torchvision.datasets.SVHN(root=config.data_dir, split='extra',
                                                 download=True, transform=transform_train)
        else:
            trainset = CustomTrainSVHN(root=config.data_dir, split='train', transform=transform_train)
            extraset = CustomTrainSVHN(root=config.data_dir, split='extra', transform=transform_train)

        total_trainset = ConcatDataset([trainset, extraset])
        testset = torchvision.datasets.SVHN(root=config.data_dir, split='test',
                                            download=True, transform=transform_test)
    elif config.dataset == 'reduced_svhn':
        if config.exp_type == 'baseline':
            total_trainset = torchvision.datasets.SVHN(root=config.data_dir, split='train',
                                                       download=True, transform=transform_train)
        else:
            total_trainset = CustomTrainSVHN(root=config.data_dir, split='train', transform=transform_train)

        # print('svhn dataset attributes: {}'.format(total_trainset.__dict__.keys()))
        # print('labels type: {}'.format(type(total_trainset.labels)))
        # print('labels[:10]: {}'.format(type(total_trainset.labels[:10])))
        # total_trainset = torchvision.datasets.CIFAR10(root=config.data_dir, train=True,
        #                                               download=True, transform=transform_train)
        # print('targets type: {}'.format(type(total_trainset.targets)))
        # print('targets[:10]: {}'.format(type(total_trainset.targets[:10])))
        # print('cifar10 dataset attributes: {}'.format(total_trainset.__dict__.keys()))
        #
        # exit()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=73257-config.sample_num, random_state=0)  # 1000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.labels)
        train_idx, valid_idx = next(sss)
        labels = [total_trainset.labels[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.labels = labels
        assert config.sample_num == len(train_idx)
        print('sample number: {}'.format(config.sample_num))
        testset = torchvision.datasets.SVHN(root=config.data_dir, split='test',
                                            download=True, transform=transform_test)
    elif config.dataset == 'imagenet':
        # raise Exception('the customized data loader is not implemented yet.')
        if config.exp_type == 'baseline':
            total_trainset = ImageNet(root=config.data_dir, split='train',
                                      transform=transform_train)
        else:
            total_trainset = CustomTrainImageNet(root=config.data_dir,
                                                 transform=transform_train)

        testset = ImageNet(root=config.data_dir, split='val', transform=transform_test)
        # exit()
        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]
    elif config.dataset == 'reduced_imagenet':
        # raise Exception('the customized data loader is not implemented yet.')
        # randomly chosen indices
        # idx120 = sorted(random.sample(list(range(1000)), k=120))
        idx120 = [16, 23, 52, 57, 76, 93, 95, 96, 99, 121, 122, 128, 148, 172, 181, 189, 202, 210, 232, 238, 257, 258, 259, 277, 283, 289, 295, 304, 307, 318, 322, 331, 337, 338, 345, 350, 361, 375, 376, 381, 388, 399, 401, 408, 424, 431, 432, 440, 447, 462, 464, 472, 483, 497, 506, 512, 530, 541, 553, 554, 557, 564, 570, 584, 612, 614, 619, 626, 631, 632, 650, 657, 658, 660, 674, 675, 680, 682, 691, 695, 699, 711, 734, 736, 741, 754, 757, 764, 769, 770, 780, 781, 787, 797, 799, 811, 822, 829, 830, 835, 837, 842, 843, 845, 873, 883, 897, 900, 902, 905, 913, 920, 925, 937, 938, 940, 941, 944, 949, 959]
        assert len(idx120) == 120
        if config.exp_type == 'baseline':
            total_trainset = ImageNet(root=config.data_dir, split='train',
                                      transform=transform_train)
        else:
            total_trainset = CustomTrainImageNet(root=config.data_dir,
                                                 transform=transform_train)

        testset = ImageNet(root=config.data_dir, split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

        # first sample 50K from 1000 classes and further sample 6400 from 128 classes
        sss = StratifiedShuffleSplit(n_splits=1, test_size=len(total_trainset) - 50000, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        print('train_idx len: {}'.format(len(train_idx)))
        print('valid_idx len: {}'.format(len(valid_idx)))

        # filter out
        train_idx = list(filter(lambda x: total_trainset.samples[x][1] in idx120, train_idx))
        valid_idx = list(filter(lambda x: total_trainset.samples[x][1] in idx120, valid_idx))
        test_idx = list(filter(lambda x: testset.samples[x][1] in idx120, range(len(testset))))

        targets = [idx120.index(total_trainset.targets[idx]) for idx in train_idx]
        # # count samples in each class
        # tmp = []
        # for c in range(120):
        #     tmp.append(targets.count(c))
        # print('sample num in each class: {}'.format(tmp))
        # print('sample sum: {}'.format(sum(tmp)))
        # print('total_trainset len: {}'.format(len(total_trainset.samples)))
        # print('train_idx len: {}'.format(len(train_idx)))
        # exit()

        for idx in range(len(total_trainset.samples)):
            if total_trainset.samples[idx][1] not in idx120:
                continue
            total_trainset.samples[idx] = (total_trainset.samples[idx][0], idx120.index(total_trainset.samples[idx][1]))
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        for idx in range(len(testset.samples)):
            if testset.samples[idx][1] not in idx120:
                continue
            testset.samples[idx] = (testset.samples[idx][0], idx120.index(testset.samples[idx][1]))
        testset = Subset(testset, test_idx)
        print('reduced_imagenet train=', len(total_trainset))
        print('reduced_imagenet test=', len(testset))
    else:
        raise ValueError('invalid dataset name: {}'.format(config.dataset))

    # if total_aug is not None and augs is not None:
    #     total_trainset.set_preaug(augs, total_aug)
    #     print('set_preaug-')

    # train_sampler = None
    # if split > 0.0:
    #     sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
    #     sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
    #     for _ in range(split_idx + 1):
    #         train_idx, valid_idx = next(sss)
    #
    #     # if target_lb >= 0:
    #     #     train_idx = [i for i in train_idx if total_trainset.targets[i] == target_lb]
    #     #     valid_idx = [i for i in valid_idx if total_trainset.targets[i] == target_lb]
    #
    #     train_sampler = SubsetRandomSampler(train_idx)
    #     valid_sampler = SubsetSampler(valid_idx)
    #
    #     # if horovod:
    #     #     import horovod.torch as hvd
    #     #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_sampler, num_replicas=hvd.size(), rank=hvd.rank())
    # else:
    #     valid_sampler = SubsetSampler([])
    #
    #     # if horovod:
    #     #     import horovod.torch as hvd
    #     #     train_sampler = torch.utils.data.distributed.DistributedSampler(valid_sampler, num_replicas=hvd.size(), rank=hvd.rank())
    print('batch size: {}'.format(config.batch_size))
    trainloader = torch.utils.data.DataLoader(total_trainset, batch_size=config.batch_size,
                                              shuffle=True, num_workers=config.workers, pin_memory=True,
                                              drop_last=True)
    # validloader = torch.utils.data.DataLoader(
    #     total_trainset, batch_size=batch, shuffle=False, num_workers=16, pin_memory=True,
    #     sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False,
                                             num_workers=config.workers, pin_memory=True, drop_last=False)
    return trainloader, testloader


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

class CustomTrainCifar10(torchvision.datasets.CIFAR10):
    def __init__(self, root, transform=None, target_transform=None):
        super(CustomTrainCifar10, self).__init__(root=root, train=True, transform=transform,
                                                 target_transform=target_transform, download=True)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img_tf = self.transform(img)

        img = self.normalize(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img, img_tf], target


class CustomTrainCifar100(torchvision.datasets.CIFAR100):
    def __init__(self, root, transform=None, target_transform=None):
        super(CustomTrainCifar100, self).__init__(root=root, train=True, transform=transform,
                                                 target_transform=target_transform, download=True)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img_tf = self.transform(img)

        img = self.normalize(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img, img_tf], target

class CustomTrainSVHN(torchvision.datasets.SVHN):
    def __init__(self, root, split, transform=None, target_transform=None):
        super(CustomTrainSVHN, self).__init__(root=root, split=split, transform=transform,
                                              target_transform=target_transform, download=True)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img_tf = self.transform(img)

        img = self.normalize(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img, img_tf], target

class CustomTrainImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root, split='train', **kwargs):
        torchvision.datasets.VisionDataset.__init__(self, root, **kwargs)
        self.split = split
        self._read_meta_file()
        print(split + ' image num: {}'.format(len(self.samples)))

        self.normalize = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_tf = self.transform(sample)

        sample = self.normalize(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [sample, sample_tf], target

    def _read_meta_file(self):

        import json
        with open(self.meta_file, 'r') as f:
            images_dict = json.load(f)

        samples = [(os.path.join(self.split_folder, img_path), label) for img_path, label in images_dict.items()]
        classes = list(set([img_path.split('/')[0] for img_path in images_dict.keys()]))
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.loader = torchvision.datasets.folder.default_loader
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @property
    def split_folder(self):
        # return os.path.join(self.root, self.split)
        if self.split == 'train':
            return os.path.join(self.root, 'train')
        elif self.split == 'val':
            return os.path.join(self.root, 'validation')
        else:
            raise Exception('unknown split: {}'.format(self.split))

    @property
    def meta_file(self):
        if self.split == 'train':
            return os.path.join(self.root, 'train.json')
        elif self.split == 'val':
            return os.path.join(self.root, 'val.json')
        else:
            raise Exception('unknown split: {}'.format(self.split))
