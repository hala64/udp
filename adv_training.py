import os
import logging
import torch.optim as optim
import time
import torchvision

from torchvision import datasets, transforms
from model.ResNet import *
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils import *
from adv_defense.trades import trades_loss
from adv_defense.utils_awp import AdvWeightPerturb
from adv_defense.mart import mart_loss
from generate_poisoned_dataset.generate_data import generate_poison

def train(model, device, train_loader, optimizer,  epoch, awp_adversary,
          epsilon=8/255, num_steps=10, step_size=-1/255, attack='None', normal=False, make_labels=False, random=False):
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if make_labels:
            for i in range(len(target)):
                target[i] = (target[i] + 1) % 10

        if attack == 'PGD':
            #data, target = Variable(data, requires_grad=True), Variable(target)
            data = PGD_attack(model, data, target, device, epsilon=epsilon,
                num_steps=num_steps, step_size=step_size, normal=normal,random=random)
        elif attack == 'CW':
            data = CW_attack(model, data, target, device, epsilon=epsilon,
                num_steps=num_steps, step_size=step_size, normal=normal,random=random)
        elif attack == 'AWP':
            data = PGD_attack(model, data, target, device, epsilon=epsilon,
                num_steps=num_steps, step_size=step_size, normal=normal,random=random)

        model.train()
        if attack == 'AWP':
            awp = awp_adversary.calc_awp(inputs_adv=data, targets=target)
            awp_adversary.perturb(awp)
        else:
            optimizer.zero_grad()

        if attack == 'TRADES':
            loss = trades_loss(model=model, x_natural=data, y=target, optimizer=optimizer, step_size=step_size,
                               epsilon=epsilon, perturb_steps=num_steps, beta=6.0)
        elif attack == 'MART':
            loss = mart_loss(model=model, x_natural=data, y=target, optimizer=optimizer, step_size=step_size,
                               epsilon=epsilon, perturb_steps=num_steps, beta=6.0)
        else:
            output = model(data)
            loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        if attack == 'AWP':
            awp_adversary.restore(awp)
        #torch.cuda.empty_cache()

        # print progress
        if attack != 'None':
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        else:
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

def adv_train(args):
    dir = args.adv_dir
    at_method = args.at_method
    poison_method = args.poison_method
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if not args.no_cuda else {}
    batch_size = args.batch_size
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.FileHandler(os.path.join(dir, f'{at_method}-{poison_method}.log')),logging.StreamHandler()])


    poison = generate_poison(args, poison_method)
    if args.data_augmentation:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.ToTensor()
        
    if args.dataset == 'CIFAR-10':
        poison_data = CIFAR10PoisonIndex(root=args.data, train=True, download=True, transform=transform, delta=poison)
    elif args.dataset == 'CIFAR-100':
        poison_data = CIFAR100PoisonIndex(root=args.data, train=True, download=True, transform=transform, delta=poison)

    poison_train_loader = torch.utils.data.DataLoader(poison_data, batch_size=batch_size, shuffle=True, **kwargs)
    
    _, labels, _, testset = data_utils(args)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, **kwargs)

    victim_model_name = args.victim_model
    victim_lr = args.lr
    epsilons = [1 / 255, 2 / 255, 3 / 255, 4 / 255, 6 / 255, 8 / 255, 12 / 255, 16 / 255]
          
    for iter in range(len(epsilons)):
        epsilon = epsilons[iter]
        logger.info(
            f'do adv training {at_method} for {poison_method}, '
            f'victim model is {victim_model_name}, learning rate={victim_lr},epsilon={epsilon}')

        victim_model = ResNet18(num_classes=labels).to(device)
        victim_model = torch.nn.DataParallel(victim_model).to(device)
        victim_optimizer = optim.SGD(victim_model.parameters(), lr=victim_lr,
                                     momentum=args.momentum, weight_decay=args.weight_decay)

        proxy = ResNet18(num_classes=labels).to(device)
        proxy = torch.nn.DataParallel(proxy).to(device)
        proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
        awp_adversary = AdvWeightPerturb(model=victim_model, proxy=proxy, proxy_optim=proxy_opt, gamma=0.01)

        #victim_model.load_state_dict(
        #    torch.load(os.path.join(dir, f'adv_training_victim_model_{victim_model_name}.pt'))['model'])
        #victim_optimizer.load_state_dict(
        #   torch.load(os.path.join(dir, f'adv_training_victim_model_{victim_model_name}.pt'))['optimizer'])

        start_epoch = 0
        if args.dataset == 'CIFAR-10':
            total_epoch = args.adv_epochs
            victim_schduler = MultiStepLR(optimizer=victim_optimizer, milestones=[75, 90, 100], gamma=0.1)
        elif args.dataset == 'CIFAR-100' or 'TinyImageNet':
            total_epoch = 200
            victim_schduler = MultiStepLR(optimizer=victim_optimizer, milestones=[80, 140, 180], gamma=0.1)
        for epoch in range(start_epoch, total_epoch):
            start_time = time.time()

            train(victim_model, device, poison_train_loader, victim_optimizer, epoch,
                  awp_adversary, epsilon=epsilon, num_steps=args.num_steps, step_size=epsilon/args.num_steps,
                  attack=at_method,random=True)

            train_time = time.time()
            train_loss, train_accuracy = eval_train(victim_model, device, poison_train_loader)

            test_loss, test_accuracy = eval_test(victim_model, device, test_loader)
            test_time = time.time()

            victim_schduler.step()

            logger.info('Epoch \t Train Time \t Test Time \t \t Train Loss \t Train Acc \t Test Loss \t Test Acc')

            logger.info('%d \t %.1f \t \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f ',
                        epoch + 1, train_time - start_time, test_time - train_time,
                        train_loss, train_accuracy,test_loss, test_accuracy)
            #victim_save = {'model': victim_model.state_dict(), 'optimizer': victim_optimizer.state_dict()}
            #torch.save(victim_save, os.path.join(dir, f'adv_training_victim_model_{victim_model_name}.pt'))
