import os
from generate_poisoned_dataset.args import get_args
from noise_test import poisontest
from evaluate_poison import evaluate
from poison_detection import poisondetect
from adv_training import adv_train

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args = get_args()

    #evaluate the poison power with different learning rates and victim models
    evaluate(args)

    # test different types of noise learning
    poisontest(args)

    # detect poisons
    poisondetect(args)
    
    # defend poisons by stronger data augmentations and adversarial noise
    denoise(args)
    denoise_aug(args)

    # defend poisons by adversarial training
    adv_train(args)


if __name__ == '__main__':
    main()
