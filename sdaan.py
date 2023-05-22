import torch.optim as optim
import numpy as np
import logging
import time
from torch.utils.data import dataloader,dataset
from torch.optim.lr_scheduler import MultiStepLR
from model.ResNet import ResNet18
from model.MLP import two_layer_NN
from utils import *
from PIL import Image
from generate_poisoned_dataset.generate_data import generate_poison


def denoise(args):
    dir = args.adv_aug_dir
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
        handlers=[logging.FileHandler(os.path.join(dir, f'denoise-{args.dataset}-{poison_method}.log')),
                  logging.StreamHandler()])

    poison = generate_poison(args, poison_method)

    if args.dataset == 'CIFAR-10':
        num_classes = 10
        poison_data = CIFAR10PoisonIndex(root=args.data, train=True, download=True, transform=transforms.ToTensor(), delta=poison)
    elif args.dataset == 'CIFAR-100':
        num_classes = 100
        poison_data = CIFAR100PoisonIndex(root=args.data, train=True, download=True, transform=transforms.ToTensor(),
                                         delta=poison)

    train_loader = torch.utils.data.DataLoader(poison_data, batch_size=batch_size, shuffle=True, **kwargs)

    width = 32
    victim_model = two_layer_NN(input_dims=width * width * 3, output_dims=num_classes).to(device)
    victim_optimizer = optim.SGD(victim_model.parameters(), lr=0.01, momentum=args.momentum, weight_decay=args.weight_decay)
    victim_schduler = MultiStepLR(optimizer=victim_optimizer, milestones=[75, 90], gamma=0.1)

    total_epoch = 100

    for epoch in range(total_epoch):
        start_time = time.time()

        train(victim_model, device, train_loader, victim_optimizer, epoch, attack='PGD', step_size=2/255,
              num_steps=10)
        train_time = time.time()

        if (epoch+1) % 10 == 0:
            train_loss, train_accuracy = eval_train(victim_model, device, train_loader, attack_method='PGD')
        else:
            train_loss, train_accuracy = -1., -1.
        test_time = time.time()


        victim_schduler.step()

        logger.info('Epoch \t Train Time \t Test Time \t \t Train Loss \t Train Acc ')
        logger.info('%d \t %.1f \t \t %.1f \t %.4f \t %.4f',
                    epoch+1, train_time - start_time, test_time - train_time, train_loss, train_accuracy)

        if (epoch+1) % 10 == 0:
            epsilon = torch.zeros(50000, 3, 32, 32).cuda()
            for batch_idx, (data, target, index) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                perturb_data = PGD_attack(victim_model, data, target, device)
                epsilon[index] = perturb_data - data

            torch.save(epsilon.cpu(), os.path.join('poisoned_data', f'denoise_{args.dataset}_{args.poison_method}_poisons.pt'))

def denoise_aug(args):
    dir = args.adv_aug_dir
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
        handlers=[logging.FileHandler(os.path.join(dir, f'denoise-{args.dataset}-{poison_method}-{args.strength}.log')),
                  logging.StreamHandler()])


    if args.dataset == 'CIFAR-10':
        num_classes = 10

    elif args.dataset == 'CIFAR-100':
        num_classes = 100


    poison1 = torch.load(os.path.join('poisoned_data', f'{args.dataset}_{args.poison_method}_poisons.pt')).cpu()
    poison2 = torch.load(os.path.join('poisoned_data', f'denoise_{args.dataset}_{args.poison_method}_poisons.pt')).cpu()
    poison = poison1 + poison2

    output_size = (32, 32)
    s = args.strength

    logger.info(s)

    transform_train = transforms.Compose([transforms.ToPILImage(),
        transforms.RandomResizedCrop(output_size, scale=(1 - 0.9 * s, 1.0)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomApply(
                                              [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)],
                                              p=0.8 * s),
                                          transforms.RandomGrayscale(p=0.2 * s),
                                          transforms.ToTensor()])

    if s == 0.0:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    if args.dataset == 'CIFAR-10':
        poison_data = CIFAR10PoisonIndex(root=args.data, train=True, download=True, transform=transform_train, delta=poison)

        test_set = CIFAR10PoisonIndex(root=args.data, train=False, download=True,
                                     delta=None, ratio=1.0, transform=transforms.ToTensor())
    elif args.dataset == 'CIFAR-100':
        poison_data = CIFAR100PoisonIndex(root=args.data, train=True, download=True, transform=transform_train,
                                         delta=poison)

        test_set = CIFAR100PoisonIndex(root=args.data, train=False, download=True,
                                      delta=None, ratio=1.0, transform=transforms.ToTensor())


    train_loader = torch.utils.data.DataLoader(poison_data, batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)


    victim_model = ResNet18(num_classes=num_classes).to(device)
    victim_optimizer = optim.SGD(victim_model.parameters(), lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay)
    #victim_schduler = MultiStepLR(optimizer=victim_optimizer, milestones=[75, 90], gamma=0.1)

    total_epoch = 200

    for epoch in range(total_epoch):
        start_time = time.time()

        adjust_learning_rate(victim_optimizer, epoch, total_epoch, schedule='cosine', lr=args.lr)

        train(victim_model, device, train_loader, victim_optimizer, epoch, attack='None', step_size=2/255)
        train_time = time.time()

        train_loss, train_accuracy = eval_train(victim_model, device, train_loader, attack_method='None')
        test_loss, test_accuracy = eval_test(victim_model, device, test_loader)

        test_time = time.time()

        #victim_schduler.step()

        logger.info('Epoch \t Train Time \t Test Time \t \t Train Loss \t Train Acc \t Test Loss \t Test Acc ')
        logger.info('%d \t %.1f \t \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f',
                    epoch+1, train_time - start_time, test_time - train_time,
                    train_loss, train_accuracy, test_loss, test_accuracy)

