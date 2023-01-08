from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
import logging
import time
from torchvision import datasets, transforms
import collections

from model.WideResNet import *
from model.ResNet import *
from model.VGG import *
from model.WideResNet import WideResNet
from model.MLP import MLP
from torch.autograd import Variable
#from attack import *
#from tttest import isotropic_noise
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import dataloader



batch_size = 128
torch.manual_seed(10)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #normalize
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    #normalize
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)


def PGD_attack(model, X, y, device, epsilon=8/255, num_steps=10, step_size=2/255, random=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    x_adv = X.detach()

    if random:
        random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-epsilon, epsilon).to(device)
        x_adv = Variable(x_adv.data + random_noise, requires_grad=True)

    for i in range(num_steps):
        x_adv.requires_grad_()
        # optimizer = torch.optim.SGD([X_pgd], lr=1e-3)
        # optimizer.zero_grad()

        with torch.enable_grad():
            loss = criterion(model(x_adv), y)
            # print(loss)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)  # clamp到[0,1]就不能进行normalize了
        # optimizer.step()

    adv = x_adv - X.detach()
    return adv

def rem_poison(model, dataset, device, perturbed_steps=20, step_size=1/255, epsilon=8/255, batch_size=100):
    trans_steps = 1
    noise_rate = 1.0
    poison_set = list([torch.zeros(3, 32, 32), 0] for _ in range(len(dataset)))
    model.eval()
    for i in range(int(len(trainset)/batch_size)):
        print(i)
        data = list(trainset[j][0] for j in range(i*batch_size,(i+1)*batch_size))
        target = list(torch.tensor(trainset[j][1]) for j in range(i*batch_size,(i+1)*batch_size))
        data = torch.stack(data)
        target = torch.stack(target)
        data, target = data.to(device), target.to(device)
        for _ in range(perturbed_steps):
            defense_noise = torch.rand_like(data)
            defense_noise = Variable(defense_noise.data, requires_grad=True)
            for j in range(trans_steps):
                adv_noise = list(torch.rand_like(data) for _ in range(trans_steps))
                transform_data = list(torch.rand_like(data) for _ in range(trans_steps))
                indices = np.random.permutation(len(data))[:int(len(data) * noise_rate)]
                transform_data[j] = (data + defense_noise)[indices]
                adv_noise[j] = PGD_attack(model, transform_data[j], target, device, epsilon=epsilon / 2,
                                          num_steps=perturbed_steps, step_size=step_size / 2, random=False)
            with torch.enable_grad():
                loss = 0
                for j in range(trans_steps):
                    loss += nn.CrossEntropyLoss()(model(transform_data[j] + adv_noise[j]), target)
                loss /= trans_steps
            grad = torch.autograd.grad(loss, [defense_noise])[0]
            defense_noise = defense_noise.detach() - step_size * torch.sign(grad.detach())
            data = torch.min(torch.max(data + defense_noise, data - epsilon), data + epsilon)
            data = torch.clamp(data, 0., 1.)
        indices1 = np.random.permutation(len(data))[:int(len(data) * noise_rate)]
        transform_data1 = data[indices1]
        adv_noise = PGD_attack(model, transform_data1, target, device, epsilon=epsilon / 2,
                               num_steps=perturbed_steps, step_size=step_size / 2, random=False)
        poison_data = transform_data1 + adv_noise
        poison_data = poison_data.detach().cpu()

        for k in range(len(poison_data)):
            poison_set[i * batch_size + k][0] = poison_data[k]

        for l in range(len(trainset)):
            # poison_trainset[k][0] = poison_trainloader[int(k / batch_size)][0][k % batch_size]
            poison_set[l][1] = trainset[l][1]

    return poison_set

def train(model, device, train_loader, optimizer, epoch, random='None',
          epsilon=8/255,perturbed_steps=10,step_size=2/255, attack_method='None'):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if random == 'random':
            random_noise = torch.FloatTensor(*data.shape).uniform_(-epsilon,  epsilon).to(device)
            data = data + random_noise
            data = torch.clamp(data, 0., 1.)
        elif random == 'fixed':
            data = data + torch.tensor(0.1).to(device)
            data = torch.clamp(data, 0., 1.)


        if attack_method == 'PGD':
            #data, target = Variable(data, requires_grad=True), Variable(target)
            adv_data = data + PGD_attack(model, data, target, device, epsilon=epsilon,
                              num_steps=perturbed_steps, step_size=step_size, random=False)
        elif attack_method == 'REM':
            trans_steps = 1
            noise_rate = 1.0
            for _ in range(perturbed_steps):
                defense_noise = torch.rand_like(data)
                defense_noise = Variable(defense_noise.data, requires_grad=True)
                for j in range(trans_steps):
                    adv_noise = list(torch.rand_like(data) for _ in range(trans_steps))
                    transform_data = list(torch.rand_like(data) for _ in range(trans_steps))
                    indices = np.random.permutation(len(data))[:int(len(data) * noise_rate)]
                    transform_data[j] = (data + defense_noise)[indices]
                    adv_noise[j] = PGD_attack(model, transform_data[j], target, device, epsilon=epsilon/2,
                                      num_steps=perturbed_steps, step_size=step_size/2, random=False)
                with torch.enable_grad():
                    loss = 0
                    for j in range(trans_steps):
                        loss += nn.CrossEntropyLoss()(model(transform_data[j] + adv_noise[j]),target)
                    loss /= trans_steps
                grad = torch.autograd.grad(loss, [defense_noise])[0]
                defense_noise = defense_noise.detach() - step_size * torch.sign(grad.detach())
                data = torch.min(torch.max(data + defense_noise, data - epsilon), data + epsilon)
                data = torch.clamp(data, 0., 1.)
            indices1 = np.random.permutation(len(data))[:int(len(data) * noise_rate)]
            transform_data1 = data[indices1]
            adv_noise = PGD_attack(model, transform_data1, target, device, epsilon=epsilon/2,
                                      num_steps=perturbed_steps, step_size=step_size/2, random=False)
            data = transform_data1 + adv_noise

        model.train()

        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def eval(model, device, loader, attack_method=None):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            if attack_method == 'PGD-20(8/255)':
                data = data + PGD_attack(model, data, target, device, epsilon=8 / 255, num_steps=20, step_size=1 / 255)

            output = model(data)
            loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return loss, accuracy

def main():
    # init model, ResNet18() can be also used here for training
    logger = logging.getLogger(__name__)
    model_dir = 'rem_results'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.FileHandler(os.path.join(model_dir, 'output1.log')),logging.StreamHandler()])

    source_model = ResNet18().to(device)
    source_model = torch.nn.DataParallel(source_model).to(device)

    train_model = ResNet18().to(device)
    train_model = torch.nn.DataParallel(train_model).to(device)

    source_optimizer = optim.SGD(source_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_optimizer = optim.SGD(train_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    source_scheduler = MultiStepLR(optimizer=source_optimizer, milestones=[50, 75, 90], gamma=0.1)
    train_scheduler = MultiStepLR(optimizer=train_optimizer, milestones=[50, 75, 90], gamma=0.1)

    start_epoch = 0
    poison_epoch = 100
    train_epoch = 100
    total_epoch = poison_epoch + train_epoch

    print('start source model train')
    for epoch in range(start_epoch, poison_epoch):
        start_time = time.time()
        train(source_model, device, train_loader, source_optimizer, epoch,
              epsilon=8 / 255, perturbed_steps=10, step_size=2 / 255, attack_method='REM')
        train_time = time.time()
        source_scheduler.step()

        torch.save(source_model.state_dict(), os.path.join(model_dir, 'rem_source_model.pt'))
        torch.save(source_optimizer.state_dict(), os.path.join(model_dir, 'rem-source-optim.tar'))

        logger.info('Epoch \t Train Time')

        logger.info('%d \t %.1f', epoch + 1, train_time - start_time)

    rem_poison_trainset = rem_poison(source_model, trainset, device,
                                     perturbed_steps=20, step_size=1/255, epsilon=8/255, batch_size=100)
    torch.save(rem_poison_trainset, os.path.join(model_dir, 'rem_poison_trainset'))
    rem_poison_trainset = torch.load(os.path.join(model_dir, 'rem_poison_trainset'))

    print('poison data has been created')
    rem_poison_train_loader = torch.utils.data.DataLoader(rem_poison_trainset, batch_size=batch_size, shuffle=True, **kwargs)

    print('start train model train')
    for epoch in range(np.max([0, start_epoch - poison_epoch]), total_epoch - poison_epoch):
        start_time = time.time()
        train(source_model, device, rem_poison_train_loader, train_optimizer, epoch,
              epsilon=8 / 255, perturbed_steps=5, step_size=2 / 255, attack_method='None')
        train_time = time.time()

        train_loss, train_accuracy = eval(train_model, device, rem_poison_train_loader)
        if epoch % 5 == 0:
            train_robust_loss, train_robust_accuracy = eval(train_model, device, rem_poison_train_loader, attack_method='PGD')
        else:
            train_robust_loss, train_robust_accuracy = -1, -1

        test_loss, test_accuracy = eval(train_model, device, test_loader)
        if epoch % 5 == 0:
            test_robust_loss, test_robust_accuracy = eval(train_model, device, test_loader, attack_method='PGD')
        else:
            test_robust_loss, test_robust_accuracy = -1, -1

        test_time = time.time()
        train_scheduler.step()

        logger.info('Epoch \t Train Time \t Test Time \t \t Train Loss \t Train Acc \t Train Robust Loss '
                    '\t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')

        logger.info('%d \t %.1f \t \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f',
                    epoch, train_time - start_time, test_time - train_time,
                    train_loss, train_accuracy, train_robust_loss, train_robust_accuracy,
                    test_loss, test_accuracy, test_robust_loss, test_robust_accuracy)

        torch.save(train_model.state_dict(), os.path.join(model_dir, 'rem_train_model.pt'))
        torch.save(train_optimizer.state_dict(), os.path.join(model_dir, 'rem-train-opt.tar'))

    print('end train model train')

if __name__ == '__main__':
    main()