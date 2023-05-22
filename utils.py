import torch
import numpy as np
import torchvision
import copy
import torch.nn as nn
import torch.nn.functional as F
import os

from PIL import Image
from torchvision import transforms
from generate_poisoned_dataset.TinyImageNet_load import TinyImageNet_load
from torchtoolbox.transform import Cutout
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder


class CIFAR10PoisonIndex(CIFAR10):
    def __init__(self, delta: torch.FloatTensor = None, pseudo_labels=None, ratio=1.0,
                 class_wise_pseduo_labels=None, **kwargs):
        super(CIFAR10PoisonIndex, self).__init__(**kwargs)
        self.delta = delta
        if pseudo_labels is not None:
            self.targets = pseudo_labels.astype(np.compat.long)
        assert ratio <= 1.0 and ratio > 0

        if self.delta is not None:
            if len(delta) == 10:
                if class_wise_pseduo_labels is not None:
                    class_wise_pseduo_targets = class_wise_pseduo_labels.astype(np.compat.long)
                    self.delta = self.delta[torch.tensor(class_wise_pseduo_targets)]
                else:
                    self.delta = self.delta[torch.tensor(self.targets)]
            if delta.shape != self.data.shape:
                self.delta = self.delta.permute(0, 2, 3, 1)
                assert self.delta.shape == self.data.shape

            set_size = int(len(self.data) * ratio)
            if set_size < len(self.data):
                self.delta[set_size:] = 0.0
            self.delta = self.delta.mul(255).cpu().numpy()
            self.data = np.clip(self.data.astype(np.float32) + self.delta, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, idx


class CIFAR100PoisonIndex(CIFAR100):
    def __init__(self, delta: torch.FloatTensor = None, pseudo_labels=None, ratio=1.0,
                 class_wise_pseduo_labels=None, **kwargs):
        super(CIFAR100PoisonIndex, self).__init__(**kwargs)
        self.delta = delta
        if pseudo_labels is not None:
            self.targets = pseudo_labels.astype(np.compat.long)
        assert ratio <= 1.0 and ratio > 0

        if self.delta is not None:
            if len(delta) == 100:
                if class_wise_pseduo_labels is not None:
                    class_wise_pseduo_targets = class_wise_pseduo_labels.astype(np.compat.long)
                    self.delta = self.delta[torch.tensor(class_wise_pseduo_targets)]
                else:
                    self.delta = self.delta[torch.tensor(self.targets)]
            if delta.shape != self.data.shape:
                self.delta = self.delta.permute(0, 2, 3, 1)
                assert self.delta.shape == self.data.shape

            set_size = int(len(self.data) * ratio)
            if set_size < len(self.data):
                self.delta[set_size:] = 0.0
            self.delta = self.delta.mul(255).cpu().numpy()
            self.data = np.clip(self.data.astype(np.float32) + self.delta, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, idx


class TinyImageNetPoisonIndex(ImageFolder):
    def __init__(self, root, train=True, transform=None, delta: torch.FloatTensor = None):
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        super().__init__(root, transform=transform)
        self.delta = delta

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.delta is not None:
            if len(self.delta) == 200:
                delta = self.delta[target]
            else:
                delta = self.delta[index]
            delta = delta.mul(255).numpy().transpose(1, 2, 0)
            sample = np.asarray(sample)
            sample = np.clip(sample.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            sample = Image.fromarray(sample, mode='RGB')

        sample = self.transform(sample)
        return sample, target, index



def Normalize(x, mean, std, device):
    x = x.to(device)
    for i in range(len(x)):
        for j in range(len(x[0])):
            s, m = torch.tensor(std[j]).to(device), torch.tensor(mean[j]).to(device)
            x[i][j] = (x[i][j] - m) / s
    return x

def datapoison(width,labels, model, trainset, device, num_steps=20, step_size=-1/255,
               epsilon=8/255, batch_size=100, make_labels=False):
    poison_trainset = list([torch.zeros(3, width, width), 0] for _ in range(len(trainset)))
    model.eval()
    for i in range(int(len(trainset)/batch_size)):
        print(i)
        data = list(trainset[j][0] for j in range(i*batch_size,(i+1)*batch_size))
        if make_labels:
            target = list(torch.tensor((trainset[j][1] + 3) % labels) for j in range(i * batch_size, (i + 1) * batch_size))
        else:
            target = list(torch.tensor(trainset[j][1]) for j in range(i * batch_size, (i + 1) * batch_size))

        data = torch.stack(data)
        target = torch.stack(target)
        data, target = data.to(device), target.to(device)

        poison_data = PGD_attack(model, data, target, device, epsilon=epsilon,
                   num_steps=num_steps, step_size=step_size)
        poison_data = poison_data.detach().cpu()

        for k in range(len(poison_data)):
            poison_trainset[i * batch_size + k][0] = poison_data[k]

    for l in range(len(trainset)):
        poison_trainset[l][1] = trainset[l][1]

    return poison_trainset


def mixup_data(data, target, device, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_data = lam * data + (1 - lam) * data[index, :]
    target_a, target_b = target, target[index]
    return mixed_data, target_a, target_b, lam

def mixup_criterion(criterion, pred, target_a, target_b, lam):
    return lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)

def CW(output, y):
    x_target = output[:, y]
    logit, label = torch.sort(output, dim=1)

    ind = (label[:, -1] == y).float()
    u = torch.arange(output.shape[0])

    # lossvalue = ((-output[u, y] + logit[:, -2] * ind +logit[:, -1] * (1. - ind))/\
    #            (logit[:, -1] - logit[:, -3] + 1e-12)).sum()

    lossvalue = -output[u, y] + logit[:, -2] * ind + logit[:, -1] * (1. - ind)
    return lossvalue

def quantized(x, device, smaller, larger):
    x = x.to(device)
    #x: one image for CIFAR10 is 3 * 32 * 32
    assert torch.min(x) >= 0 and torch.max(x) <= 1
    assert 0 <= smaller <= 0.5 and 0.5 <= larger <= 1
    x = torch.where(x > 0.5, larger, smaller)
    return x.cpu()

def mask(x,p=0.9):
    y = copy.deepcopy(x)
    #print(y)
    x = x.view(-1)
    length = len(x.view(-1))
    masked = torch.randperm(length)[:int(p*length)]
    x[masked] = 0
    x = x.view_as(y)
    return x

def PGD_attack(model, X, y, device, epsilon=8/255, num_steps=10, step_size=1/255, normal=False, random=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    x_adv = X.detach()# + 0.001 * torch.randn(*X.shape).to(device).detach()
    if random:
        x_adv =  X.detach() + torch.FloatTensor(*X.shape).uniform_(-epsilon, epsilon).to(device).detach()

    for _ in range(num_steps):
        optimizer = torch.optim.SGD([x_adv], lr=0.1)
        optimizer.zero_grad()
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = criterion(model(x_adv), y)
        grad = torch.autograd.grad(loss, [x_adv],only_inputs=True)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        if normal:
            normal_epsilon = Normalize(epsilon * torch.ones(len(x_adv), 3, 32, 32),
                                       mean=[0., 0., 0.], std=[0.2023, 0.1994, 0.2010], device=device)
            x_adv = torch.min(torch.max(x_adv, X - normal_epsilon), X + normal_epsilon)
            normal_zero = Normalize(torch.zeros(len(x_adv), 3, 32, 32),
                                       mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], device=device)
            normal_one = Normalize(torch.ones(len(x_adv), 3, 32, 32),
                                       mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], device=device)
            x_adv = torch.clamp(x_adv, normal_zero, normal_one)
        else:
            x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)  # clamp to [0,1] without normalize
        optimizer.step()
    #torch.cuda.empty_cache()
    return x_adv

def CW_attack(model, X, y, device, epsilon=8/255, num_steps=10, step_size=1/255, normal=False, random=False):
    model.eval()
    x_adv = X.detach()# + 0.001 * torch.randn(*X.shape).to(device).detach()
    if random:
        x_adv =  X.detach() + torch.FloatTensor(*X.shape).uniform_(-epsilon, epsilon).to(device).detach()

    for _ in range(num_steps):
        #optimizer = torch.optim.SGD([X_pgd], lr=1e-3)
        #optimizer.zero_grad()
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = CW(model(x_adv), y).mean()
        grad = torch.autograd.grad(loss, [x_adv],only_inputs=True)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        if normal:
            normal_epsilon = Normalize(epsilon * torch.ones(len(x_adv), 3, 32, 32),
                                       mean=[0., 0., 0.], std=[0.2023, 0.1994, 0.2010], device=device)
            x_adv = torch.min(torch.max(x_adv, X - normal_epsilon), X + normal_epsilon)
            normal_zero = Normalize(torch.zeros(len(x_adv), 3, 32, 32),
                                       mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], device=device)
            normal_one = Normalize(torch.ones(len(x_adv), 3, 32, 32),
                                       mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], device=device)
            x_adv = torch.clamp(x_adv, normal_zero, normal_one)
        else:
            x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)  # clamp to [0,1] without normalize
    #torch.cuda.empty_cache()
    return x_adv

def train(model, device, train_loader, optimizer, epoch, epsilon=8/255, num_steps=10, step_size=-1/255, attack='None',
          normal=False, make_labels=False, mixup=False, quantize=False, masked=False, random=False):
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if make_labels:
            target = (target + 1) % 10

        if attack == 'PGD':
            #data, target = Variable(data, requires_grad=True), Variable(target)
            data = PGD_attack(model, data, target, device, epsilon=epsilon,
                num_steps=num_steps, step_size=step_size, normal=normal,random=random)
        elif attack == 'CW':
            data = CW_attack(model, data, target, device, epsilon=epsilon,
                num_steps=num_steps, step_size=step_size, normal=normal,random=random)

        model.train()
        optimizer.zero_grad()

        if mixup:
            mixed_data, target_a, target_b, lam = mixup_data(data, target, device, alpha=1.0)
            output = model(mixed_data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        elif quantize:
            quantized_data = quantized(data,device,0.25,0.75).to(device)
            small_data = torch.clamp(data-0.9,0,1)
            big_data =torch.clamp(data+0.9,0,1)
            output1 = model(quantized_data)
            output2 = model(small_data)
            output3 = model(big_data)
            output4 = model(data)
            loss = criterion(output1,target) + criterion(output2,target) + \
                   criterion(output3,target) + criterion(output4,target)
        elif masked:
            data = mask(data,p=0.7)
            output = model(data)
            loss = criterion(output, target)
        else:
            output = model(data)
            loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        #torch.cuda.empty_cache()

        # print progress
        if attack == 'PGD' or attack == 'CW':
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        else:
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))



def eval_train(model, device, train_loader,attack_method='None',normal=False):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in train_loader:
            data, target = data.to(device), target.to(device)
            if attack_method == 'PGD':
                data = PGD_attack(model, data, target, device, epsilon=8 / 255, num_steps=20, step_size=1 / 255,
                                  normal=normal)
            elif attack_method == 'CW':
                data = CW_attack(model, data, target, device, epsilon=8 / 255,
                                 num_steps=20, step_size=1 / 255, normal=normal)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_accuracy = correct / len(train_loader.dataset)
    return train_loss, train_accuracy


def eval_test(model, device, test_loader,attack_method='None',normal=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            if attack_method == 'PGD':
                data = PGD_attack(model, data, target, device, epsilon=8 / 255, num_steps=20, step_size=1 / 255,
                                  normal=normal)
            elif attack_method == 'CW':
                data = CW_attack(model, data, target, device, epsilon=8 / 255,
                                 num_steps=20, step_size=1 / 255, normal=normal)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def adjust_learning_rate(optimizer, epoch,total_epoch,lr=0.1,schedule='cosine'):
    if schedule == 'cosine':
        lr = lr * 0.5 * (1 + np.cos(epoch / total_epoch * np.pi))
    elif schedule == 'piecewise':
        if epoch >= 75:
            lr = lr * 0.1
        if epoch >= 90:
            lr = lr * 0.01
        if epoch >= 100:
            lr = lr * 0.001
    elif schedule == 'adv_piecewise':
        if epoch >= 100:
            lr = lr * 0.1
        if epoch >= 150:
            lr = lr * 0.01
    elif schedule == 'vit_piecewise':
        if epoch == 0:
            lr = lr * 0.1
        if epoch >= 75:
            lr = lr * 0.1
        if epoch >= 90:
            lr = lr * 0.01
        if epoch >= 100:
            lr = lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_labels(model, trainset, device, batch_size=100):
    #nextcycle for label，to generate poison data
    dummy_trainloader = list([torch.zeros(batch_size,3, 32, 32), torch.zeros(batch_size)] for _ in range(int(len(trainset)/batch_size)))
    dummy_trainset = list([torch.zeros(3, 32, 32), 0] for _ in range(len(trainset)))
    model.eval()
    for i in range(int(len(trainset)/batch_size)):
        #print(i)
        dummy_target = list(torch.tensor((trainset[j][1] + 1) % 10) for j in range(i*batch_size,(i+1)*batch_size))
        #print(target)
        dummy_target = torch.stack(dummy_target)
        #target = target.unsqueeze(0)
        dummy_target = dummy_target.to(device)

        dummy_target = dummy_target.detach().cpu()
        #print(target)
        dummy_trainloader[i][1] = dummy_target
    #print(poison_trainset[11])

    for k in range(len(trainset)):
        #print(poison_trainset[k])
        dummy_trainset[k][0] = trainset[k][0]
        dummy_trainset[k][1] = dummy_trainloader[int(k / batch_size)][1][k % batch_size].item()
        #print(poison_trainset[k])

    return dummy_trainset

def data_utils(args):
    if args.dataset == 'CIFAR-10':
        width = 32
        labels = 10
        if args.data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            trainset = CIFAR10PoisonIndex(root='../data', train=True, download=True,
                                                transform=transform_train)
        else:
            trainset = CIFAR10PoisonIndex(root='../data', train=True, download=True,
                                                    transform=transforms.ToTensor())
        testset = CIFAR10PoisonIndex(root='../data', train=False, download=True,
                                                transform=transforms.ToTensor())
    elif args.dataset == 'CIFAR-100':
        width = 32
        labels = 100
        if args.data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            trainset = CIFAR100PoisonIndex(root=args.data, train=True, download=True,
                                                transform=transform_train)
        else:
            trainset = CIFAR100PoisonIndex(root=args.data, train=False, download=True,
                                                    transform=transforms.ToTensor())
        testset = CIFAR100PoisonIndex(root=args.data, train=False, download=True,
                                                transform=transforms.ToTensor())
    elif args.dataset == 'TinyImageNet':
        width = 64
        labels = 200
        if args.data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                Cutout(),
                transforms.ToTensor(),
            ])
            trainset = TinyImageNetPoisonIndex(root='../data/tiny-imagenet-200/', train=True,
                                        transform=transform_train)
        else:
            trainset = TinyImageNetPoisonIndex(root='../data/tiny-imagenet-200/', train=True,
                                        transform=transforms.ToTensor())
        testset = TinyImageNetPoisonIndex(root='../data/tiny-imagenet-200/', train=False,
                                        transform=transforms.ToTensor())
    else:
        raise {'dataset error'}

    return width, labels, trainset, testset

def poison_aug(args, poison_set):
    if args.dataset == 'CIFAR-10':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        clean_set = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transforms.ToTensor())
        poison = torch.zeros(len(poison_set), 3, 32, 32)
        assert len(clean_set) == len(poison)
        for i in range(len(poison)):
            poison[i] = poison_set[i][0] - clean_set[i][0]
        poison_set = CIFAR10PoisonIndex(root=args.data, train=True, download=True, transform=transform, delta=poison)
    if args.dataset == 'CIFAR-100':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        clean_set = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transforms.ToTensor())
        poison = torch.zeros(len(poison_set), 3, 32, 32)
        assert len(clean_set) == len(poison)
        for i in range(len(poison)):
            poison[i] = poison_set[i][0] - clean_set[i][0]
        poison_set = CIFAR100PoisonIndex(root=args.data, train=True, download=True, transform=transform, delta=poison)
    elif args.dataset == 'TinyImageNet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            Cutout(),
            transforms.ToTensor(),
        ])
        clean_set = TinyImageNet_load(root='../data/tiny-imagenet-200/', train=True,
                                     transform=transforms.ToTensor())
        poison = torch.zeros(len(poison_set), 3, 64, 64)
        assert len(clean_set) == len(poison)
        for i in range(len(poison)):
            poison[i] = poison_set[i][0] - clean_set[i][0]
        poison_set = TinyImageNetPoisonIndex(root='../data/tiny-imagenet-200/', train=True, transform=transform, delta=poison)
    else:
        raise {'dataset error'}

    return poison_set

class AdversarialPoison(torch.utils.data.Dataset):
    def __init__(self, root, baseset):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.listdir(os.path.join(root, 'data'))
        self.root = root

        # Load images into memory to prevent IO from disk
        self.data, self.targets = self._load_images()

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.targets[idx]

    def _load_images(self):
        data = list(torch.zeros(3,32,32) for _ in range(len(self.baseset)))
        targets = list(0 for _ in range(len(self.baseset)))
        num_data_to_load = len(self.baseset)
        for i in range(num_data_to_load):
            true_index = int(self.samples[i].split('.')[0])
            _, label = self.baseset[true_index]
            data[true_index]= (Image.open(os.path.join(self.root, 'data', self.samples[i])).copy())
            targets[true_index] = (label)
        return data, targets

