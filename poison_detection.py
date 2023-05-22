import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
import logging
import time
import os
from torchvision import datasets, transforms
from torch.utils.data import dataloader,dataset
from torch.optim.lr_scheduler import MultiStepLR
from model.ResNet import ResNet18
from model.VGG import VGG16
from model.MLP import three_layer_NN, Linear,two_layer_NN
from utils import *
from PIL import Image
from generate_poisoned_dataset.generate_data import generate_poison


def poisondetect(args):
    dir = args.detect_dir
    detection_method = args.detection_method
    poison_method = args.poison_method
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if not args.no_cuda else {}
    batch_size = args.batch_size
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.FileHandler(os.path.join(dir, f'detection-{detection_method}-{poison_method}.log')),
                  logging.StreamHandler()])


    victim_model_name = args.victim_model
    victim_lr = args.detect_lr
    logger.info(f'do poison detection {detection_method} for {poison_method}, '
        f'victim model is {victim_model_name}, learning rate={victim_lr}')


    poison = generate_poison(args, poison_method)
    if args.data_augmentation and args.dataset != 'TinyImageNet':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    elif args.data_augmentation and args.dataset == 'TinyImageNet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            Cutout(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.ToTensor()

    if args.dataset == 'CIFAR-10':
        poison_data = CIFAR10PoisonIndex(root=args.data, train=True, download=True, transform=transform, delta=poison)
    elif args.dataset == 'CIFAR-100':
        poison_data = CIFAR100PoisonIndex(root=args.data, train=True, download=True, transform=transform, delta=poison)
    elif args.dataset == 'TinyImageNet':
        poison_data = TinyImageNetPoisonIndex(root=args.data, train=True, transform=transform, delta=poison)
   
    width, _, _, _ = data_utils(args.dataset)

    torch.manual_seed(args.seed)
    if args.dataset == 'CIFAR-10' or 'CIFAR-100':
        train_set, val_set = dataset.random_split(poison_data, (40000, 10000))
    elif args.dataset == 'TinyImageNet':
        train_set, val_set = dataset.random_split(poison_data, (90000, 10000))
    else:
        raise {'dataset error'}

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True,**kwargs)

    if args.detection_method == 'bias-0.5':
        train_set = list([train_set[i][0] - 0.5 * torch.ones(1), train_set[i][1]] for i in range(len(train_set)))
    elif args.detection_method == 'bias+0.5':
        train_set = list([train_set[i][0] + 0.5 * torch.ones(1), train_set[i][1]] for i in range(len(train_set)))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,**kwargs)

    if args.detection_method == 'bias-0.5' or 'bias+0.5':
        victim_model = ResNet18().to(device)
    elif args.detection_method == 'simple-linear':
        victim_model = Linear(input_dims=width*width*3).to(device)
    elif args.detection_method == 'simple-2NN':
        victim_model = two_layer_NN(input_dims=width*width*3).to(device)
    else:
        raise {'detection method error'}

    victim_model = torch.nn.DataParallel(victim_model).to(device)
    victim_optimizer = optim.SGD(victim_model.parameters(), lr=victim_lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)

    if args.dataset == 'CIFAR-10':
        total_epoch = args.epochs
        victim_schduler = MultiStepLR(optimizer=victim_optimizer, milestones=[75, 90], gamma=0.1)
    elif args.dataset == 'CIFAR-100' or 'TinyImageNet':
        total_epoch = 200
        victim_schduler = MultiStepLR(optimizer=victim_optimizer, milestones=[60, 120, 160], gamma=0.1)

    if args.detection_method == 'bias+0.5' or 'bias-0.5':
        victim_lr = args.lr

    for epoch in range(total_epoch):
        start_time = time.time()

        train(victim_model, device, train_loader, victim_optimizer, epoch, attack='None')
        train_time = time.time()

        train_loss, train_accuracy = eval_train(victim_model, device, train_loader)
        val_loss, val_accuracy = eval_test(victim_model, device, val_loader)
        test_time = time.time()

        victim_schduler.step()

        logger.info('Epoch \t Train Time \t Test Time \t \t Train Loss \t Train Acc \t Test Loss \t Test Acc ')
        logger.info('%d \t %.1f \t \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f',
                    epoch+1, train_time - start_time, test_time - train_time,
                    train_loss, train_accuracy, val_loss, val_accuracy)
