import torch
import torchvision
import os
from torchvision import transforms
from torch.utils.data import dataloader
from PIL import Image
from utils import *

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

def autoregressive(args):
    model_dir = './ar-results'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    torch.manual_seed(1111)

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.ToTensor())

    poison_trainset = AdversarialPoison(os.path.join(args.poison_dir, 'linf-8-cifar10-ar', trainset))
    poisons = torch.zeros(len(poison_trainset), 3, 32, 32)
    
    for i in range(len(poisons)):
        poisons[i] = poison_trainset[i][0].cpu()
               
    return poisons
