import torch
import numpy as np
import os

from PIL import Image
from torchvision import transforms
from .TinyImageNet_load import TinyImageNet_load
from torchtoolbox.transform import Cutout
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder


class CIFAR10PoisonIndex(CIFAR10):
    def __init__(self, delta: torch.FloatTensor = None, pseudo_labels=None, ratio=1.0,
                 class_wise_pseduo_labels=None, **kwargs):
        super(CIFAR10PoisonIndex, self).__init__(**kwargs)
        self.delta = delta
        if pseudo_labels is not None:
            self.targets = pseudo_labels.astype(np.compat.long)
        assert ratio <= 1.0 and ratio > 0

        if self.delta is not None:
            if len(delta) == 10:
                if class_wise_pseduo_labels is not None:
                    class_wise_pseduo_targets = class_wise_pseduo_labels.astype(np.compat.long)
                    self.delta = self.delta[torch.tensor(class_wise_pseduo_targets)]
                else:
                    self.delta = self.delta[torch.tensor(self.targets)]
            if delta.shape != self.data.shape:
                self.delta = self.delta.permute(0, 2, 3, 1)
                assert self.delta.shape == self.data.shape

            set_size = int(len(self.data) * ratio)
            if set_size < len(self.data):
                self.delta[set_size:] = 0.0
            self.delta = self.delta.mul(255).cpu().numpy()
            self.data = np.clip(self.data.astype(np.float32) + self.delta, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, idx


class CIFAR100PoisonIndex(CIFAR100):
    def __init__(self, delta: torch.FloatTensor = None, pseudo_labels=None, ratio=1.0,
                 class_wise_pseduo_labels=None, **kwargs):
        super(CIFAR100PoisonIndex, self).__init__(**kwargs)
        self.delta = delta
        if pseudo_labels is not None:
            self.targets = pseudo_labels.astype(np.compat.long)
        assert ratio <= 1.0 and ratio > 0

        if self.delta is not None:
            if len(delta) == 100:
                if class_wise_pseduo_labels is not None:
                    class_wise_pseduo_targets = class_wise_pseduo_labels.astype(np.compat.long)
                    self.delta = self.delta[torch.tensor(class_wise_pseduo_targets)]
                else:
                    self.delta = self.delta[torch.tensor(self.targets)]
            if delta.shape != self.data.shape:
                self.delta = self.delta.permute(0, 2, 3, 1)
                assert self.delta.shape == self.data.shape

            set_size = int(len(self.data) * ratio)
            if set_size < len(self.data):
                self.delta[set_size:] = 0.0
            self.delta = self.delta.mul(255).cpu().numpy()
            self.data = np.clip(self.data.astype(np.float32) + self.delta, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, idx


class TinyImageNetPoisonIndex(ImageFolder):
    def __init__(self, root, train=True, transform=None, delta: torch.FloatTensor = None):
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        super().__init__(root, transform=transform)
        self.delta = delta

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.delta is not None:
            if len(self.delta) == 200:
                delta = self.delta[target]
            else:
                delta = self.delta[index]
            delta = delta.mul(255).numpy().transpose(1, 2, 0)
            sample = np.asarray(sample)
            sample = np.clip(sample.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            sample = Image.fromarray(sample, mode='RGB')

        sample = self.transform(sample)
        return sample, target, index
    

def data_utils(args):
    if args.dataset == 'CIFAR-10':
        width = 32
        labels = 10
        if args.data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            trainset = CIFAR10PoisonIndex(root='../data', train=True, download=True,
                                                transform=transform_train)
        else:
            trainset = CIFAR10PoisonIndex(root='../data', train=True, download=True,
                                                    transform=transforms.ToTensor())
        testset = CIFAR10PoisonIndex(root='../data', train=False, download=True,
                                                transform=transforms.ToTensor())
    elif args.dataset == 'CIFAR-100':
        width = 32
        labels = 100
        if args.data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            trainset = CIFAR100PoisonIndex(root=args.data, train=True, download=True,
                                                transform=transform_train)
        else:
            trainset = CIFAR100PoisonIndex(root=args.data, train=False, download=True,
                                                    transform=transforms.ToTensor())
        testset = CIFAR100PoisonIndex(root=args.data, train=False, download=True,
                                                transform=transforms.ToTensor())
    elif args.dataset == 'TinyImageNet':
        width = 64
        labels = 200
        if args.data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                Cutout(),
                transforms.ToTensor(),
            ])
            trainset = TinyImageNetPoisonIndex(root='../data/tiny-imagenet-200/', train=True,
                                        transform=transform_train)
        else:
            trainset = TinyImageNetPoisonIndex(root='../data/tiny-imagenet-200/', train=True,
                                        transform=transforms.ToTensor())
        testset = TinyImageNetPoisonIndex(root='../data/tiny-imagenet-200/', train=False,
                                        transform=transforms.ToTensor())
    else:
        raise {'dataset error'}

    return width, labels, trainset, testset
