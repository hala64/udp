# Is Unlearnable Data Poisoning Really Unlearnable: Explanations, Detections, and Defenses of Unlearnable Data Poisoning Attacks

# Install requirements
```shell
pip install -r requirements.txt
```

# Quick Starts

```bash
python main.py \
    --dataset CIFAR-10 \
    --victim-model ResNet18 \
    --batch-size 128 \
    --seed 1111 \
    --lr 0.1 \
    --epochs 100 \
    -- log-interval 10 \
    --data-augmentation \
    --poison-method region4 \
    --poison-dir  ./poisoned_data \
    --evaluate-dir ./unlearnable_evaluation \
    --noise-test-dir ./noise_learning_test \
    --detection-method simple-linear \
    --detect-dir ./unlearnable_detection \
    --at-method PGD \
    --epsilon 8/255 \
    --num-steps 10 \
    --step-size 2/255 \
    --adv-dir ./unlearnable_defense \
```
