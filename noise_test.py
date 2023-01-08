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
from model.ResNet import ResNet18
from model.VGG import VGG16
from model.MLP import three_layer_NN
from utils import *
from generate_poisoned_dataset.generate_data import generate_poison


def poisontest(args):
    dir = args.noise_test_dir
    poison_method = args.poison_method
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if not args.no_cuda else {}
    batch_size = args.batch_size
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.FileHandler(os.path.join(dir, f'noise-learning-test-{poison_method}.log')),logging.StreamHandler()])

    victim_model_name = args.victim_model
    victim_lr = args.lr
    logger.info(f'do noise learning test for {poison_method}, victim model is {victim_model_name},'
                f'learning rate={victim_lr}')

    poison_data = generate_poison(args, poison_method)
    clean_data = data_utils(args.dataset)

    if args.dataset == 'CIFAR-10' or 'CIFAR-100':
        torch.manual_seed(args.seed)
        trainset, valset = dataset.random_split(clean_data, (40000, 10000))
        torch.manual_seed(args.seed)
        poison_trainset, poison_valset = dataset.random_split(poison_data, (40000, 10000))
    elif args.dataset == 'TinyImageNet':
        torch.manual_seed(args.seed)
        trainset, valset = dataset.random_split(clean_data, (40000, 10000))
        torch.manual_seed(args.seed)
        poison_trainset, poison_valset = dataset.random_split(poison_data, (90000, 10000))
    else:
        raise {'dataset error'}


    testset1 = [tuple([poison_valset[i][0] - valset[i][0], valset[i][1]]) for i in range(len(valset))]
    #torch.save(testset1, os.path.join('./poison_test_data', 'errmax_poison_testset_1'))
    #testset1 = torch.load(os.path.join('./poison_test_data', 'errmax_poison_testset_1')) #D_te(1)

    testset2 = [tuple([poison_valset[i][0] - valset[i][0] + 0.3 * torch.ones(1), valset[i][1]]) for i in range(len(valset))]
    #torch.save(testset2, os.path.join('./poison_test_data', 'errmax_poison_testset_2'))
    #testset2 = torch.load(os.path.join('./poison_test_data', 'errmax_poison_testset_2')) #D_te(2)

    testset3 = [tuple([poison_valset[i][0] - valset[i][0] + 0.5 * torch.ones(1), valset[i][1]]) for i in range(len(valset))]
    #torch.save(testset3, os.path.join('./poison_test_data', 'errmax_poison_testset_3'))
    #testset3 = torch.load(os.path.join('./poison_test_data', 'errmax_poison_testset_3')) #D_te(3)

    testset4 = [tuple([poison_valset[i][0] - valset[i][0] + 0.7 * torch.ones(1), valset[i][1]]) for i in range(len(valset))]
    #torch.save(testset4, os.path.join('./poison_test_data', 'errmax_poison_testset_4'))
    #testset4 = torch.load(os.path.join('./poison_test_data', 'errmax_poison_testset_4')) #D_te(4)

    if args.dataset == 'CIFAR-10':
        sorted_valset = SortedCIFAR10(valset)
        sorted_poison_valset = SortedCIFAR10(poison_valset)
    elif args.dataset == 'CIFAR-100':
        sorted_valset = SortedCIFAR100(valset)
        sorted_poison_valset = SortedCIFAR100(poison_valset)
    elif args.dataset == 'CIFAR-10':
        sorted_valset = SortedTinyImageNet(valset)
        sorted_poison_valset = SortedTinyImageNet(poison_valset)

    testset5 = [tuple([sorted_poison_valset[i][0] - sorted_valset[i][0] + sorted_valset
    [((i+3000)%(len(sorted_valset)))][0], sorted_valset[i][1]])for i in range(len(sorted_valset))]
    #torch.save(testset5, os.path.join('./poison_test_data', 'errmax_poison_testset_5'))
    #testset5 = torch.load(os.path.join('./poison_test_data', 'errmax_poison_testset_5')) #D_te(5)

    testset6 = [tuple([sorted_poison_valset[i][0] - sorted_valset[i][0] + sorted_valset[(i+3000)%(len(sorted_valset))][0],
                       sorted_valset[(i+3000)%(len(sorted_valset))][1]]) for i in range(len(sorted_valset))]
    #torch.save(testset6, os.path.join('./poison_test_data', 'errmax_poison_testset_6'))
    #testset6 = torch.load(os.path.join('./poison_test_data', 'errmax_poison_testset_6')) #D_te(6)

    testset7 = [tuple([poison_valset[i][0] - valset[i][0] + 32/255 * torch.sign(torch.randn(3,32,32)),
                       valset[i][1]]) for i in range(len(valset))]
    #torch.save(testset7, os.path.join('./poison_test_data', 'errmax_poison_testset_7'))
    #testset7 = torch.load(os.path.join('./poison_test_data', 'errmax_poison_testset_7')) #D_te(7)

    testset8 = [tuple([poison_valset[i][0] - valset[i][0] + torch.Tensor(3, 32, 32).uniform_(-32/255, 32/255),
                       valset[i][1]]) for i in range(len(valset))]
    #torch.save(testset8, os.path.join('./poison_test_data', 'errmax_poison_testset_8'))
    #testset8 = torch.load(os.path.join('./poison_test_data', 'errmax_poison_testset_8')) #D_te(8)

    poison_train_loader = torch.utils.data.DataLoader(poison_trainset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, **kwargs)
    poison_test_loader = torch.utils.data.DataLoader(poison_valset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader1 = torch.utils.data.DataLoader(testset1, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader2 = torch.utils.data.DataLoader(testset2, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader3 = torch.utils.data.DataLoader(testset3, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader4 = torch.utils.data.DataLoader(testset4, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader5 = torch.utils.data.DataLoader(testset5, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader6 = torch.utils.data.DataLoader(testset6, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader7 = torch.utils.data.DataLoader(testset7, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader8 = torch.utils.data.DataLoader(testset8, batch_size=batch_size, shuffle=False, **kwargs)


    victim_model = ResNet18().to(device)
    victim_model = torch.nn.DataParallel(victim_model).to(device)
    victim_optimizer = optim.SGD(victim_model.parameters(), lr=victim_lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)
    victim_epoch = args.epochs
    for epoch in range(victim_epoch):
        start_time = time.time()
        adjust_learning_rate(victim_optimizer, epoch, victim_epoch, schedule='piecewise', lr=victim_lr)

        train(victim_model, device, poison_train_loader, victim_optimizer, epoch, attack='None')
        train_time = time.time()

        _, train_accuracy = eval_train(victim_model, device, poison_train_loader)
        _, test_accuracy = eval_test(victim_model, device, test_loader)
        _, test_accuracy0 = eval_test(victim_model, device, poison_test_loader)
        _, test_accuracy1 = eval_test(victim_model, device, test_loader1)
        _, test_accuracy2 = eval_test(victim_model, device, test_loader2)
        _, test_accuracy3 = eval_test(victim_model, device, test_loader3)
        _, test_accuracy4 = eval_test(victim_model, device, test_loader4)
        _, test_accuracy5 = eval_test(victim_model, device, test_loader5)
        _, test_accuracy6 = eval_test(victim_model, device, test_loader6)
        _, test_accuracy7 = eval_test(victim_model, device, test_loader7)
        _, test_accuracy8 = eval_test(victim_model, device, test_loader8)
        test_time = time.time()

        logger.info('Epoch \t Train Time \t Test Time \t Train Loss \t Train Acc \t Test Loss \t Test Acc '
                    '\t Test acc1 \t Test acc2\t Test acc3 \t Test acc4 \t Test acc5 \t Test acc6'
                    '\t Test acc7 \t Test acc8')
        logger.info('%d \t %.1f \t \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f\t %.4f '
                    '\t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f'
                    , epoch + 1, train_time - start_time, test_time - train_time, train_accuracy,
                    test_accuracy,  test_accuracy0, test_accuracy1, test_accuracy2,test_accuracy3, test_accuracy4,
                    test_accuracy5, test_accuracy6, test_accuracy7, test_accuracy8)