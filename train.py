import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics, callbacks
from argparse import ArgumentParser
import os
import math
from src.models import ParNet
from src.utils import load_dataset


LR = 1e-2


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset',
                        default='imagenet')
    parser.add_argument('--scale',
                        default='small')
    parser.add_argument('--batch-size',
                      default=8,
                      type=int)
    parser.add_argument('--gpus',
                        help='gpus id',
                        default='0')
    parser.add_argument('--output',
                      help='Path to save checkpoint',
                      default='workdirs')
    parser.add_argument('--epochs',
                      default=100,
                      type=int)
    args = parser.parse_args()
    return args


def scheduler(epoch, lr):
    if epoch in [30, 60, 80]:
        return 0.1 * lr
    else:
        return lr


def main():
    # setup multi gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    num_gpus = len(args.gpus.split(','))
    devices = ["GPU:{}".format(i) for i in range(num_gpus)]
    strategy = tf.distribute.MirroredStrategy(devices)
    global_batch_size = args.batch_size * num_gpus

    # create output dir if not exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

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
    train_ds, val_ds = load_dataset(dataset=args.dataset,
                                    num_classes=num_classes,
                                    augmentation=False,
                                    train=True,
                                    batch_size=args.batch_size,
                                    image_shape=input_shape)

    # optimizer, loss
    optimizer = optimizers.SGD(learning_rate=LR)
    loss = losses.CategoricalCrossentropy()

    with strategy.scope():
        accuracy = metrics.CategoricalAccuracy(name='acc')
        model = ParNet(dataset=args.dataset,
                       scale=args.scale,
                       num_classes=num_classes,
                       train=True,
                       input_shape=input_shape)
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[accuracy])
        model.summary()

    # checkpoint
    filepath = os.path.join(args.output,
                            "saved-model-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5")
    checkpoint = callbacks.ModelCheckpoint(filepath,
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True)
    lr_scheduler = callbacks.LearningRateScheduler(scheduler, verbose=1)

    # fit
    model.fit(train_ds, epochs=args.epochs, batch_size=global_batch_size,
              callbacks=[checkpoint, lr_scheduler], validation_data=val_ds)


if __name__ == '__main__':
    args = parse_args()
    main()