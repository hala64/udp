import torch
import os
import logging
import torch.optim as optim
import time

from torch.utils.data import DataLoader
from .model.ResNet import ResNet18
from .utils import *


def adv_induce(args):
    dataset = args.dataset
    model_dir = args.poison_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if not args.no_cuda else {}
    batch_size = args.batch_size
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.FileHandler(os.path.join(model_dir, 'adv_inducing.log')),logging.StreamHandler()])

    width, labels, trainset = data_utils(dataset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)


    source_model = ResNet18(num_classes=labels).to(device)
    source_model = torch.nn.DataParallel(source_model).to(device)
    source_optimizer = optim.SGD(source_model.parameters(), lr=args.lr, momentum=args.momentum,
                                 weight_decay=args.weight_decay)

    start_epoch = 0
    poison_epoch = 120
    print('start source model train')
    for epoch in range(start_epoch, poison_epoch):
        start_time = time.time()
        adjust_learning_rate(source_optimizer, epoch, poison_epoch, schedule='piecewise')
        # AT训练source model
        train(source_model, device, train_loader, source_optimizer, epoch,
              epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size,attack='PGD',make_labels=True)
        train_time = time.time()
        source_save = {'model': source_model.state_dict(), 'optimizer': source_optimizer.state_dict()}
        torch.save(source_save, os.path.join(model_dir, 'source_model_advinduce.pt'))

        logger.info('Epoch \t Train Time')

        logger.info('%d \t %.1f', epoch, train_time - start_time)

    source_model.load_state_dict(torch.load(os.path.join(model_dir, 'source_model_advinduce.pt'))['model'])
    poison_trainset = datapoison(width, labels, source_model, trainset, device,
                                 num_steps=60,step_size=0.5/255,make_labels=True)

    torch.save(poison_trainset,os.path.join((model_dir, 'adv_inducing_poison_trainset.pth'))

    return poison_trainset
