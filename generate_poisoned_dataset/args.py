import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Generate poisoned data')
    parser.add_argument('--dataset', default='CIFAR-10', type=str, help='type of poisoned dataset')
    parser.add_argument('--data', type=str, default='data', help='directory of data set')

    parser.add_argument('--batch-size', type=int, default=128, help='input batch size')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--adv-epochs', type=int, default=110, help='adversarial training epochs')

    parser.add_argument('--data-augmentation', type=bool, default=True,
                        help='whether using data augmentation')

    parser.add_argument('--strength', type=float, default=1.0, help='strong augmentation strength')

    parser.add_argument('--poison-method', type=str, default='errmin',
                        help='unlearnable data poisoning method')
    parser.add_argument('--at-method', type=str, default='PGD',
                        help='adversarial training method')
    parser.add_argument('--detection-method', type=str, default='simple-2NN',
                        help='detection method(simple-linear, simple-2NN, bias-0.5 and bias+0.5)')

    parser.add_argument('--victim-model', type=str, default='ResNet18',
                        help='type of victim model')

    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--detect-lr', type=float, default=0.01, help='learning rate for detection')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA')

    parser.add_argument('--epsilon', default=8/255, help='perturbation')
    parser.add_argument('--num-steps', default=10, help='perturb step number')
    parser.add_argument('--step-size', default=1/255, help='perturb step size')

    parser.add_argument('--seed', type=int, default=1111,  help='random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate-dir', default='./unlearnable_evaluation',
                        help='directory of poison power evaluation')
    parser.add_argument('--poison-dir', default='./poisoned_data',
                        help='directory of  saved poisoned data')
    parser.add_argument('--adv-dir', default='./unlearnable_defense',
                        help='directory of adv defense')
    parser.add_argument('--detect-dir', default='./unlearnable_detection',
                        help='directory of detection')
    parser.add_argument('--noise-test-dir', default='./noise_learning_test',
                        help='directory of noise learnability')
    parser.add_argument('--adv-aug-dir', default='./unlearnable_defense_aug',
                        help='directory of aug defense')

    parser.add_argument('--save-freq', '-s', default=20, type=int, help='save frequency')

    parser.add_argument('--gpu-id', type=str, default='0')

    parser.add_argument('--evaluation', action='store_true', help='evaluate poison power')
    parser.add_argument('--noise-test', action='store_true', help='test noise learnabilities')
    parser.add_argument('--detection', action='store_true', help='detect poisons')
    parser.add_argument('--adv-training', action='store_true', help='defend poisons by adversarial training')
    parser.add_argument('--strong-aug', action='store_true', help='defend poisons by stronger augmentations')

    parser.add_argument('--multi-steps', action='store_true', help='if using multi steps when generating adv noises')

    return parser.parse_args()
