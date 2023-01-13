import torch
import torch.optim as optim
import logging
import time
import os
from torch.utils.data import dataloader
from model.ResNet import ResNet18
from model.VGG import VGG16
from utils import *


def class_wise_poison(args, width, labels, sorted_poison_trainset,sorted_trainset):
    poison_trainset = [[torch.zeros(3, width, width), 0] for _ in range(len(sorted_trainset))]
    classwise_noise = [torch.zeros(3, width, width) for _ in range(labels)]
    for k in range(10):
        noise = [sorted_poison_trainset[i][0]-sorted_trainset[i][0]
                 for i in range(k * int(len(sorted_trainset) / labels), (k + 1) * int(len(sorted_trainset) / labels))]
        noise = torch.stack(noise)
        classwise_noise[k] = torch.clamp(torch.sum(noise, dim=0), -args.epsilon,args.epsilon)
        #print(classwise_noise[k])

    for j in range(len(sorted_trainset)):
        data, label = sorted_trainset[j]
        #print(data.size())
        #print(classwise_noise[label].size())
        poison_trainset[j][0] = data + classwise_noise[label]
        poison_trainset[j][0] = torch.clamp(poison_trainset[j][0],0.,1.)
        poison_trainset[j][1] = sorted_trainset[j][1]
    #print(isotropic_noise[0])
    return poison_trainset


def errmin(args, source_model_name='ResNet18', cl_wise=False):
    dataset = args.dataset
    model_dir = args.poison_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if not args.no_cuda else {}

    batch_size = args.batch_size
    width, labels, trainset, _ = data_utils(dataset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.FileHandler(os.path.join(model_dir, 'errmin.log')),logging.StreamHandler()])


    if source_model_name == 'ResNet18':
        source_model = ResNet18(num_classes=labels).to(device)
    elif source_model_name == 'VGG16':
        source_model = VGG16(num_classes=labels).to(device)
    source_model = torch.nn.DataParallel(source_model).to(device)
    source_optimizer = optim.SGD(source_model.parameters(), lr=0.025,
                                 momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = 0
    poison_epoch = 60

    logger.info('start source model train')

    for epoch in range(start_epoch, poison_epoch):
        start_time = time.time()
        adjust_learning_rate(source_optimizer, epoch, poison_epoch, schedule='cosine',lr=0.025)

        train(source_model, device, train_loader, source_optimizer, epoch, num_steps=10,step_size=-1.6/255, attack='PGD')
        train_time = time.time()

        logger.info('Epoch \t Train Time')
        logger.info('%d \t %.1f', epoch, train_time - start_time)


        source_save = {'model': source_model.state_dict(), 'optimizer': source_optimizer.state_dict()}
        torch.save(source_save, os.path.join(model_dir, f'source_model_{source_model_name}.pt'))


    logger.info('end source model train')

    source_model.load_state_dict(torch.load( os.path.join(model_dir, f'source_model_{source_model_name}.pt'))['model'])


    poison_trainset = datapoison(width, labels, source_model, trainset, device,
                                 num_steps=20, step_size=-0.8/255, epsilon=args.epsilon)
    if cl_wise:
        sorted_trainset = SortedCIFAR10(trainset).data()
        sorted_poison_trainset = SortedCIFAR10(poison_trainset).data()
        poison_trainset = class_wise_poison(args, width, labels, sorted_poison_trainset, sorted_trainset)
    if not os.path.exists('./poison_data'):
        os.makedirs('./poison_data')
    torch.save(poison_trainset, os.path.join(model_dir, f'errmin_poisoned_trainset_{source_model_name}.pth'))

    return poison_trainset
