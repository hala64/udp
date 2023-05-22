import os
import logging
import torch.optim as optim
import time
import torchvision

from torchvision import datasets, transforms
from model.ResNet import *
from model.VGG import VGG16
from model.WideResNet import WideResNet
from model.DenseNet import densenet121
from model.ViT import ViT
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils import *
from adv_defense.trades import trades_loss
from adv_defense.utils_awp import AdvWeightPerturb
from adv_defense.mart import mart_loss
from generate_poisoned_dataset.generate_data import generate_poison


def evaluate(args):
    dir = args.evaluate_dir
    poison_method = args.poison_method
    if not os.path.exists(dir):
        os.makedirs(dir)

    device = torch.device("cuda" if not args.no_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if not args.no_cuda else {}
    batch_size = args.batch_size
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.FileHandler(os.path.join(dir, f'evaluate-{poison_method}.log')),logging.StreamHandler()])


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

    poison_train_loader = torch.utils.data.DataLoader(poison_data, batch_size=batch_size, shuffle=True, **kwargs)

    width, labels, _, testset = data_utils(args)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, **kwargs)

    victim_model_name = args.victim_model
    victim_lrs = [0.1, 0.01, 0.001, 0.0001]
    if victim_model_name == 'ResNet18':
        victim_model = ResNet18(num_classes=labels).to(device)
    elif victim_model_name == 'VGG16':
        victim_model = VGG16(num_classes=labels).to(device)
    elif victim_model_name == 'WRN34-10':
        victim_model = WideResNet(num_classes=labels).to(device)
    elif victim_model_name == 'DenseNet121':
        victim_model = densenet121(num_classes=labels).to(device)
    elif victim_model_name == 'ViT':
        victim_model = ViT(image_size=width, patch_size=4, num_classes=labels, dim=512, depth=6, heads=8,
                  mlp_dim=512).to(device)
    else:
        raise {'victim model error'}
    victim_model = torch.nn.DataParallel(victim_model).to(device)

    for iter in range(len(victim_lrs)):
        victim_lr = victim_lrs[iter]
        logger.info(
            f'evaluate with learning rate={victim_lr} for {poison_method}, victim model is {victim_model_name}')

        victim_optimizer = optim.SGD(victim_model.parameters(), lr=victim_lr,
                                     momentum=args.momentum, weight_decay=args.weight_decay)

        start_epoch = 0
        if args.dataset == 'CIFAR-10':
            total_epoch = args.epochs
            victim_schduler = MultiStepLR(optimizer=victim_optimizer, milestones=[75, 90], gamma=0.1)
        elif args.dataset == 'CIFAR-100' or 'TinyImageNet':
            total_epoch = 200
            victim_schduler = MultiStepLR(optimizer=victim_optimizer, milestones=[60, 120, 160], gamma=0.1)
        for epoch in range(start_epoch, total_epoch):
            start_time = time.time()

            train(victim_model, device, poison_train_loader, victim_optimizer, epoch, attack='None',random=True)

            train_time = time.time()
            train_loss, train_accuracy = eval_train(victim_model, device, poison_train_loader)

            test_loss, test_accuracy = eval_test(victim_model, device, test_loader)
            test_time = time.time()

            victim_schduler.step()

            logger.info('Epoch \t Train Time \t Test Time \t \t Train Loss \t Train Acc \t Test Loss \t Test Acc')

            logger.info('%d \t %.1f \t \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f ',
                        epoch + 1, train_time - start_time, test_time - train_time,
                        train_loss, train_accuracy,test_loss, test_accuracy)

