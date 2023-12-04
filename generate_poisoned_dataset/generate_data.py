from .class_wise import classwise
import torch
import os

full_poison = ['random-c','region4','region16','region64', 'err-min-s', 'err-min-c', 'ntga', 'ar',
          'robustem',' hyp', 'entf', 'clean']


def generate_poison(args, poison):
    if poison == 'random-c' or poison == 'region4' or poison == 'region16' or poison == 'region64':
        poisons = classwise(args, poison)
    elif poison == 'err-min-s':
        # generated from https://github.com/HanxunH/Unlearnable-Examples
        poisons = torch.load(os.path.join('poisoned_data', 'errmins_poisons.pt'))
    elif poison == 'err-min-c':
        # generated from https://github.com/HanxunH/Unlearnable-Examples
        poisons = torch.load(os.path.join('poisoned_data', 'errminc_poisons.pt'))
    elif poison == 'ntga':
        # generated from https://github.com/lionelmessi6410/ntga
        poisons = torch.load(os.path.join('poisoned_data', 'ntga_poisons.pt'))
    elif poison == 'robustem':
        # generated from https://github.com/fshp971/robust-unlearnable-examples
        poisons = torch.load(os.path.join('poisoned_data', 'robustem_poisons.pt'))
    elif poison == 'hyp':
        # generated from https://github.com/TLMichael/Hypocritical-Perturbation
        poisons = torch.load(os.path.join('poisoned_data', 'hyp_poisons.pt'))
    elif poison == 'ar':
        # generated from https://github.com/psandovalsegura/autoregressive-poisoning
        poisons = torch.load(os.path.join('poisoned_data', 'ar_poisons.pt'))
    elif poison == 'entf':
        # generated from https://github.com/WenRuiUSTC/EntF
        poisons = torch.load(os.path.join('poisoned_data', 'entf_poisons.pt'))
    elif poison == 'clean':
        if args.dataset == 'CIFAR-10' or 'CIFAR-100':
           poisons = torch.zeros(50000, 3, 32, 32)
        elif args.dataset == 'TinyImageNet':
           poisons = torch.zeros(100000, 3, 64, 64)
    else:
        raise {'poison method error'}

    return poisons
