import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
from argparse import ArgumentParser
import os
from src.models import ParNet
from src.utils import load_dataset
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset',
                        default='imagenet')
    parser.add_argument('--scale',
                        default='small')
    parser.add_argument('--gpu',
                        help='gpus id',
                        default='0')
    parser.add_argument('--batch-size',
                        default=8,
                        type=int)
    parser.add_argument('--checkpoint',
                        help='Path to save checkpoint',
                        default='workdirs')
    args = parser.parse_args()
    return args


def main():
    # setup multi gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # number of classes
    if args.dataset == 'imagenet':
        num_classes = 1000
        input_shape = (224, 224, 3)
    elif args.dataset == 'cifar10':
        num_classes = 10
        input_shape = (32, 32, 3)
    else:
        num_classes = 100
        input_shape = (32, 32, 3)

    # load ready-to-use dataset
    test_ds = load_dataset(dataset=args.dataset,
                           num_classes=num_classes,
                           augmentation=False,
                           train=False,
                           batch_size=args.batch_size,
                           image_shape=input_shape)
    model = ParNet(dataset=args.dataset,
                   scale=args.scale,
                   num_classes=num_classes,
                   train=False,
                   input_shape=input_shape)
    model.load_weights(args.checkpoint)

    predictions = []
    gts = []
    for example in tqdm(test_ds):
        gt = example[1].numpy()
        gt = np.argmax(gt, axis=-1)
        gts.append(gt)
        pred = model(example[0], training=False).numpy()
        pred = np.argmax(pred, axis=-1)
        predictions.append(pred)
    predictions = np.concatenate(predictions, axis=0)
    gts = np.concatenate(gts, axis=0)
    print(classification_report(gts, predictions))


if __name__ == '__main__':
    args = parse_args()
    main()
