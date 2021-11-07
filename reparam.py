import tensorflow as tf
from argparse import ArgumentParser
import os
import math
from src.models import ParNet
from src.utils import parameterize


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset',
                        default='imagenet')
    parser.add_argument('--scale',
                        default='small')
    parser.add_argument('--checkpoint',
                        help='path to checkpoint')
    parser.add_argument('--output',
                        help='output file name')
    parser.add_argument('--gpu',
                        default='0')
    args = parser.parse_args()
    return args


def main():
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

    # load original model
    org_model = ParNet(dataset=args.dataset,
                       scale=args.scale,
                       num_classes=num_classes,
                       train=True,
                       input_shape=input_shape)
    org_model.load_weights(args.checkpoint)

    # create new model
    new_model = ParNet(dataset=args.dataset,
                       scale=args.scale,
                       num_classes=num_classes,
                       train=False,
                       input_shape=input_shape)

    new_model = parameterize(org_model, new_model)
    new_model.save_weights(args.output + '.hdf5')
    print('Completed !')


if __name__ == '__main__':
    args = parse_args()
    main()