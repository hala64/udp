import os
import argparse
import numpy as onp
import tensorflow as tf
from utils import *
from tqdm import tqdm

# Plotting
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
sns.set_style(style='white')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.linewidth'] = 3

parser = argparse.ArgumentParser(description="Evaluate and plot the learning curve!")
parser.add_argument("--model_type", required=True, type=str, help="Available target model:\n\
                    `fnn`, `fnn_relu`, `cnn`, `resnet18`, `resnet34`, or `densenet121`")
parser.add_argument("--dataset", required=True, type=str, help="clean dataset. `mnist`, `cifar10`, \
                    and `imagenet` are available. To use different dataset, please modify the path \
                    in the code directly")
parser.add_argument("--dtype", required=True, type=str, help="`Clean` or `NTGA`, used for figure's title")
parser.add_argument("--x_train_path", default=None, type=str, help="path for training data. Leave it empty \
                    to evaluate the performance on clean data(mnist or cifar10)")
parser.add_argument("--y_train_path", default=None, type=str, help="path for training labels. Leave it empty \
                    to evaluate the performance on clean data(mnist or cifar10)")
parser.add_argument("--x_val_path", default=None, type=str, help="path for validation data. Please specify \
                    the path for the ImageNet dataset")
parser.add_argument("--y_val_path", default=None, type=str, help="path for validation label. Please specify \
                    the path for the ImageNet dataset")
parser.add_argument("--x_test_path", default=None, type=str, help="path for testing data. The ground truth \
                    (y_test) is hidden. You can submit the prediction to Kaggle competition.")
parser.add_argument("--epoch", default=50, type=int, help="training epochs")
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--save_path", default="", type=str, help="path to save figures")
parser.add_argument("--cuda_visible_devices", default="0", type=str, help="specify which GPU to run \
                    an application on")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
seed = 0

if args.dataset == "mnist":
    num_classes = 10
    image_size = 28
    image_size_aug = 28
    lr_sgd = 1e-3
elif args.dataset == 'CIFAR-10':
    num_classes = 10
    image_size = 32
    image_size_aug = 36
    lr_sgd = 1e-3
    lr_schedule = 1e-1
elif args.dataset == 'CIFAR-100':
    num_classes = 100
    image_size = 32
    image_size_aug = 36
    lr_sgd = 1e-3
    lr_schedule = 1e-1
elif args.dataset == "TinyImageNet":
    num_classes = 200
    image_size = 64
    image_size_aug = 72
    lr_sgd = 1e-3
    lr_schedule = 1e-3
pad_size = int((image_size_aug-image_size)/2)

# Enable data augmentation for ResNet and DenseNet
if args.model_type in ["fnn", "fnn_relu", "cnn"]:
    lr_schedule = False
    augment = False
else:
    augment = True

def plot_learning_curve(train_acc, test_acc, ts, metric, dtype, save=True):
    plt.plot(ts, train_acc, label='Train', color=colors[0], linewidth=5)
    plt.plot(ts, test_acc, label='Test', color=colors[1], linewidth=5)
    plt.ylim(0, 1)
    plt.legend()
    format_plot('{:s}({:s})'.format(metric, args.dtype), 'Step')
    finalize_plot((1.25, 1))
    if save:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        plt.savefig(fname='{:s}figure_{:s}_{:s}_{:s}_{:s}.pdf'.format(args.save_path, args.dataset, 
                                                                      metric.lower(), args.model_type, 
                                                                      dtype.lower()), 
                    format="pdf", bbox_inches='tight')
    plt.show()

def augment_fn(images, labels):
    images = tf.image.pad_to_bounding_box(images, pad_size, pad_size, image_size_aug, image_size_aug)
    images = tf.image.random_crop(images, (image_size, image_size, 3))
    images = tf.image.random_flip_left_right(images)
    return images, labels

def dataset_generator(images, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(images)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

class Model():
    def __init__(self, input_shape, num_classes, model_type, lr_schedule=False, step_per_epoch=None):
        # For ResNet and DenseNet, the first few layers are different for small input, 
        # e.g. MNIST and CIFAR-10
        if model_type == 'fnn':
            from models.dnn import DNN
            self.model = DNN(input_shape, num_classes)
        elif model_type == 'fnn_relu':
            from models.dnn_relu import DNN_ReLU
            self.model = DNN_ReLU(input_shape, num_classes)
        elif model_type == 'cnn':
            from models.cnn import CNN
            self.model = CNN(input_shape, num_classes)
        elif model_type == 'resnet18':
            from models.resnet import ResNet18
            self.model = ResNet18(input_shape, num_classes)
        elif model_type == 'resnet34':
            from models.resnet import ResNet34
            self.model = ResNet34(input_shape, num_classes)
        elif model_type == 'resnet50':
            from models.resnet import ResNet50
            self.model = ResNet50(input_shape, num_classes)
        elif model_type == 'resnet101':
            from models.resnet import ResNet101
            self.model = ResNet101(input_shape, num_classes)
        elif model_type == 'resnet152':
            from models.resnet import ResNet152
            self.model = ResNet152(input_shape, num_classes)
        elif model_type == 'densenet121':
            from models.densenet import DenseNet121
            self.model = DenseNet121(input_shape, num_classes)
        else:
            print("{:s} is currently not support.".format(model_type))
        
        # Optimizer and loss function
        if model_type == 'fnn':
            self.optimizer = tf.keras.optimizers.SGD(lr_sgd)
            self.loss_object = tf.keras.losses.MeanSquaredError()
        else:
            if lr_schedule:
                learning_rate_fn = tf.keras.experimental.CosineDecay(lr_schedule, args.epoch*step_per_epoch)
                self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
            else:
                self.optimizer = tf.keras.optimizers.Adam()
#                 self.optimizer = tf.keras.optimizers.SGD(lr_sgd)
            self.loss_object = tf.keras.losses.CategoricalCrossentropy()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        
    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        
    @tf.function
    def inference_step(self, images):
        predictions = self.model(images, training=False)
        return predictions
    
    def train(self, epoch, train_ds, test_ds):
        train_acc = []
        train_l = []
        test_acc = []
        test_l = []

        for e in tqdm(range(epoch)):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for images, labels in train_ds:
                self.train_step(images, labels)

            for test_images, test_labels in test_ds:
                self.test_step(test_images, test_labels)

            template = 'Epoch {:0}, Loss: {:.4f}, Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'
            print (template.format(e+1,
                                   self.train_loss.result(),
                                   self.train_accuracy.result()*100,
                                   self.test_loss.result(),
                                   self.test_accuracy.result()*100))

            # Record results
            train_acc.append(self.train_accuracy.result())
            train_l.append(self.train_loss.result())
            test_acc.append(self.test_accuracy.result())
            test_l.append(self.test_loss.result())
            
        return train_acc, train_l, test_acc, test_l
    
    def inference(self, test_ds):
        predictions = []
        
        for test_images in test_ds:
            predictions.append(self.inference_step(test_images).numpy())
            
        predictions = onp.concatenate(predictions)
        return predictions
    
def main():
    # Prepare dataset
    print("Loading dataset...")
    if args.dataset == "MNIST":
        train_size = 50000
    elif args.dataset == "CIFAR-10" or 'CIFAR-100':
        train_size = 40000
    elif args.dataset == "TinyImageNet":
        train_size = 90000
    else:
        raise ValueError("To load custom dataset, please modify the code directly.")
        
    if args.x_train_path and args.y_train_path:
        x_train = onp.load(args.x_train_path)
        y_train = onp.load(args.y_train_path)
        x_val = onp.load(args.x_val_path)
        y_val = onp.load(args.y_val_path)
    else:
        x_train_all, y_train_all, _, _ = tuple(onp.asarray(x) for x in get_dataset(args.dataset, None, None))
        x_train_all, y_train_all = shaffle(x_train_all, y_train_all, seed)
        x_train = x_train_all[:train_size]
        y_train = y_train_all[:train_size]
        x_val = x_train_all[train_size:]
        y_val = y_train_all[train_size:]
        
    if args.x_test_path:
        x_test = onp.load(args.x_test_path)
        
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    
    if args.model_type in ["fnn", "fnn_relu"]:
        # Reshape input data into [width, height, channel] for CNNs
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        if args.x_test_path:
            x_test = x_test.reshape(x_test.shape[0], -1)
        input_shape = (x_train.shape[-1],)
        
    if augment:
        mean = onp.mean(x_train, axis=(0, 1, 2))
        std = onp.std(x_train, axis=(0, 1, 2))
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        train_ds = dataset_generator(x_train, y_train, args.batch_size)
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).\
            shuffle(train_size).batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).\
        batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    if args.x_test_path:
        if augment:
            x_test = (x_test - mean) / std
        test_ds = tf.data.Dataset.from_tensor_slices(x_test).\
            batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print("Building model...")
    step_per_epoch = int(len(x_train)/args.batch_size)
    model = Model(input_shape, num_classes, args.model_type, lr_schedule, step_per_epoch)
    
    print("Training...")
    train_acc, train_l, val_acc, val_l = model.train(args.epoch, train_ds, val_ds)
    
    ts = onp.arange(1, args.epoch+1, 1)
    plot_learning_curve(train_acc, val_acc, ts, "Accuracy", args.dtype)
#     plot_learning_curve(train_l, val_l, ts, "Loss", args.dtype)

    if args.x_test_path:
        import pandas as pd
        print("Testing...")
        y_pred = model.inference(test_ds)
        y_pred = onp.argmax(y_pred, axis=-1)
        
        # write output file
        id = [i for i in range(len(y_pred))]
        result = pd.DataFrame({'id': id, 'label': y_pred})
        result.to_csv("y_pred_{:s}.csv".format(args.dataset), index=False)
    print("================== DONE ==================")
    
if __name__ == "__main__":
    main()
