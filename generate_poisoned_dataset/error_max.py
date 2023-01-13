import torch
import torch.optim as optim
import logging
import time
import os
from torch.utils.data import dataloader
from model.ResNet import ResNet18
from model.VGG import VGG16
from utils import *


def errmax(args, source_model_name='ResNet18'):
    dataset = args.dataset
    model_dir = args.poison_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
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
        handlers=[logging.FileHandler(os.path.join(model_dir, 'errmax.log')),logging.StreamHandler()])

    if source_model_name == 'ResNet18':
        source_model = ResNet18(num_classes=labels).to(device)
    elif source_model_name == 'VGG16':
        source_model = VGG16(num_classes=labels).to(device)
    source_model = torch.nn.DataParallel(source_model).to(device)
    source_optimizer = optim.SGD(source_model.parameters(), lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = 0
    poison_epoch = 40

    logger.info('start source model train')
    start_time = time.time()
    for epoch in range(start_epoch, poison_epoch):
        start_time = time.time()
        adjust_learning_rate(source_optimizer, epoch, poison_epoch, schedule='cosine')

        train(source_model, device, train_loader, source_optimizer, epoch, num_steps=args.num_steps,
              step_size=args.step_size, attack='None')
        train_time = time.time()

        logger.info('Epoch \t Train Time')
        logger.info('%d \t %.1f', epoch, train_time - start_time)


        source_save = {'model': source_model.state_dict(), 'optimizer': source_optimizer.state_dict()}
        torch.save(source_save, os.path.join(model_dir, f'source_model_{source_model_name}.pt'))


    logger.info('end source model train')

    poison_trainset = datapoison(width, labels, source_model, trainset, device,
                                 num_steps=250,step_size=-0.064/255,make_labels=True)
    if not os.path.exists('./poison_data'):
        os.makedirs('./poison_data')
    torch.save(poison_trainset, os.path.join(model_dir, f'errmax_poison_trainset_{source_model_name}.pth'))

    end_time = time.time()
    #print(end_time-start_time)
    logger.info('poison data has been created')

    return poison_trainset
