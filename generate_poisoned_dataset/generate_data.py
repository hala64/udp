from .class_wise import classwise
from .error_min import errmin
from args import get_args

full_poison = ['random-c','region4','region16','region64','err-max','err-min-s','err-min-c','ntga','ar',
          'robustem','advinduce','hyp']


def generate_poison(args, poison):
    if poison == 'random-c' or 'region4' or 'region16'or 'region64':
        poison_trainset = classwise(args, poison)
    elif poison == 'err-min-s':
        poison_trainset = errmin(args)
    elif poison == 'err-min-c':
        poison_trainset = errmin(args, cl_wise=True)
    elif poison == 'err-max':
        from .error_max import errmax
        poison_trainset = errmax(args)
    elif poison == 'ntga':
        from .ntga.generate_attack import _ntga
        poison_trainset = _ntga(args)
    elif poison == 'robustem':
        from .robustem.generate_robust_em import robust_em
        poison_trainset = robust_em(args)
    elif poison == 'hyp':
        from .hypocritical.main import hypocritical
        poison_trainset = hypocritical()
    elif poison == 'advinduce':
        from .adv_inducing import adv_induce
        poison_trainset = adv_induce(args)
    elif poison == 'ar':
        from .ar import autoregressive
        poison_trainset = autoregressive()

    return poison_trainset