import torch
import os
import torchvision
import logging
import time
import re

from torchvision import transforms

from utils import *


def class_wise_poison(args,width,labels,trainset):
    torch.manual_seed(args.seed)
    isotropic_noise = list(torch.Tensor(3, width, width).uniform_(-8 / 255, 8 / 255) for _ in range(labels))
    poison_trainset = list([torch.zeros(3, width, width), 0] for _ in range(len(trainset)))
    for j in range(len(trainset)):
        data, label = trainset[j]
        poison_trainset[j][0] = data + isotropic_noise[label]
        poison_trainset[j][1] = label
        poison_trainset[j][0] = torch.clamp(poison_trainset[j][0],0.,1.)
    #print(isotropic_noise[0])
    return poison_trainset

def region(args, width, labels, trainset, device, patch=4, epsilon=8/255):
    sq_patch = int(np.sqrt(patch))
    batch_size = int(len(trainset) / labels)
    poison_trainset = list([torch.zeros(3, width, width), 0] for _ in range(len(trainset)))
    torch.manual_seed(args.seed)
    for i in range(int(len(trainset)/batch_size)):
        print(i)
        data = list(trainset[j][0] for j in range(int(i*batch_size),int((i+1)*batch_size)))
        data = torch.stack(data)
        data = data.to(device)
        data = data.permute(2, 3, 0, 1)
        #region_noise = list(torch.zeros(100,3,1,1) for _ in range(patch))
        for j1 in range(sq_patch):
            for j2 in range(sq_patch):
                region_noise = epsilon * torch.sign(torch.randn(1,1,1,3)).to(device)
                data[j1*int(width/sq_patch):(j1+1)*int(width/sq_patch),
                j2*int(width/sq_patch):(j2+1)*int(width/sq_patch),:,:] += \
                    region_noise.repeat(int(width/sq_patch),int(width/sq_patch),batch_size,1)

        data = data.permute(2,3,0,1)
        data = torch.clamp(data,0.,1.)

        for k in range(len(data)):
            poison_trainset[i * batch_size + k][0] = data[k].detach().cpu()

    for l in range(len(trainset)):
        poison_trainset[l][1] = trainset[l][1]

    return poison_trainset

def classwise(args, poison):
    dataset = args.dataset
    model_dir = args.poison_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    torch.manual_seed(1111)
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if not args.no_cuda else {}
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.FileHandler(os.path.join(model_dir, 'vit-random.log')),logging.StreamHandler()])


    width, labels, trainset, _ = data_utils(dataset)

    if poison == 'randomc':
        start_time =time.time()
        poison_trainset = class_wise_poison(args, width, labels, trainset)
        end_time = time.time()
        logger.info(f'generate {poison} in {end_time-start_time} seconds')
        torch.save(poison_trainset,os.path.join(model_dir, f'{poison}_trainset.pth'))

    elif poison == 'region4' or 'region16' or 'region64':
        patch = int(''.join(re.findall('[0-9]',poison)))
        start_time =time.time()
        if dataset == 'CIFAR-10':
            sorted_trainset = SortedCIFAR10(trainset).data()
        elif dataset == 'CIFAR-100':
            sorted_trainset = SortedCIFAR100(trainset).data()
        elif dataset == 'TinyImageNet':
            sorted_trainset = SortedTinyImageNet(trainset).data()
        poison_trainset = region(args, width, labels, sorted_trainset, device, patch=patch)
        end_time = time.time()
        logger.info(f'generate {poison} in {end_time-start_time} seconds')

        torch.save(poison_trainset,os.path.join(model_dir, f'{poison}_trainset.pth'))


    logger.info(f'poison data {poison} has been created')

    return poison_trainset
