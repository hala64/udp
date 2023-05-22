import os
from generate_poisoned_dataset.args import get_args
from evaluate_poison import evaluate
from poison_detection import poisondetect
from adv_training import adv_train
from sdaan import denoise, denoise_aug

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.evaluation:
        # evaluate the poison power with different learning rates and victim models
        evaluate(args)

    if args.noise_test:
        # test different types of noise learning
        poisontest(args)

    if args.detection:
        # detect poisons
        poisondetect(args)

    if args.strong_aug:
        # defend poisons by stronger data augmentations and adversarial noise
        denoise(args)
        denoise_aug(args)

    if args.adv_training:
        # defend poisons by adv training
        adv_train(args)

if __name__ == '__main__':
    main()
