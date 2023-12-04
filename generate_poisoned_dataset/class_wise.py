import torch
import os
import torchvision
import logging
import time
import numpy as np
import re
from torchvision import transforms

from utils import *


def class_wise_poison(args, target, labels=10, width=32):
    torch.manual_seed(args.seed)
    noise = torch.Tensor(labels, 3, width, width).uniform_(-args.epsilon, args.epsilon)
    return noise[target]



def region_poison(args, data, target, patch=16, epsilon=8 / 255, num_classes=10, width=32):
    torch.manual_seed(args.seed)
    tmp = copy.deepcopy(data)
    data = data.permute(0, 2, 3, 1)
    # region_noise = list(torch.zeros(100,3,1,1) for _ in range(patch))
    region_noise = epsilon * torch.sign(torch.randn(num_classes, int(np.sqrt(patch)), int(np.sqrt(patch)), 1, 1, 3))
    for i in range(len(data)):
        noise = region_noise[target[i]]
        for j1 in range(int(np.sqrt(patch))):
            for j2 in range(int(np.sqrt(patch))):
                # print(noise[j1, j2].repeat(int(32 / np.sqrt(patch)),int(32 / np.sqrt(patch)), 1).size())
                data[i, j1 * int(width / np.sqrt(patch)):(j1 + 1) * int(width / np.sqrt(patch)),
                j2 * int(width / np.sqrt(patch)):(j2 + 1) * int(width / np.sqrt(patch)), :] \
                    += noise[j1, j2].repeat(int(width / np.sqrt(patch)), int(width / np.sqrt(patch)), 1)

    data = data.permute(0, 3, 1, 2)
    data = torch.clamp(data, 0., 1.)

    return data - tmp

def classwise(args, poison):
    dataset = args.dataset
    model_dir = args.poison_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if not args.no_cuda else {}
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.FileHandler(os.path.join(model_dir, 'vit-random.log')),logging.StreamHandler()])


    width, labels, trainset, _ = data_utils(args)
    poisons = torch.zeros(len(trainset), 3, width, width)

    if poison == 'randomc':
        start_time = time.time()

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

        for data, target, index in trainloader:
            poisons[index] = class_wise_poison(args, target, labels, width=width)

        end_time = time.time()
        logger.info(f'generate {poison} in {end_time-start_time} seconds')

    elif poison == 'region4' or poison == 'region16' or poison == 'region64':
        patch = ''.join(re.findall('[0-9]', poison))
        patch = int(patch)
        start_time = time.time()

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,**kwargs)

        for data, target, index in trainloader:
            poisons[index] = region_poison(args, data, target, patch=patch, width=width, epsilon=args.epsilon)

        end_time = time.time()
        logger.info(f'generate {poison} in {end_time-start_time} seconds')


    logger.info(f'poison data {poison} has been created')

    return poisons
