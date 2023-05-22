# Detection and Defense of Unlearnable Examples

Install requirements
```shell
pip install -r requirements.txt
```

Evaluate the poison power

```bash
python main.py --evaluation --poison-method region16 --dataset CIFAR-10
```

Detection by simple networks

```bash
python main.py --detection --detection-method simple-2NN --poison-method region16 --dataset CIFAR-10
```

# Detection by bias-shifting noises

```bash
python main.py --detection --detection-method bias-0.5 --poison-method region16 --dataset CIFAR-10
```

# Defense by stronger data augmentations with adversarial noises

```bash
python main.py --strong-aug --strength 1.0 --poison-method region16 --dataset CIFAR-10
```

# Evaluation of criteria under adversarial training

```bash
python main.py --adv-training --poison-method region16 --dataset CIFAR-10
```

# Acknowledgment
Unlearnable Examples:
https://github.com/HanxunH/Unlearnable-Examples

Robust Unlearnable Examples:
https://github.com/fshp971/robust-unlearnable-examples

NTGA Poisons:
https://github.com/lionelmessi6410/ntga

AutoRegressive Poisons:
https://github.com/psandovalsegura/autoregressive-poisoning

Hypocritical Poisons:
https://github.com/TLMichael/Hypocritical-Perturbation

EntF Poisons:
https://github.com/WenRuiUSTC/EntF
