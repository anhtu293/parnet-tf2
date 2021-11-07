import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


EPSILON = 0.001


def load_dataset(dataset,
                 num_classes,
                 augmentation=False,
                 train=True,
                 batch_size=8,
                 image_shape=(224, 224, 3)):
    # normalize
    def normalize(img, label):
        img = tf.cast(img, tf.float32) / 255.
        label = tf.one_hot(label, num_classes)
        return img, label

    # data augmentation
    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_crop(img, (20, 20, 3))
        img = tf.image.resize(img, image_shape)
        return img, label

    if train:
        if dataset == 'imagenet':
            split = ['train', 'val']
        else:
            split = ['train[:90%]', 'train[90%:]']
        train, val = tfds.load(dataset, split=split,
                               shuffle_files=True,
                               as_supervised=True)
        train = train.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        val = val.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        if augmentation:
            train = train.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            val = val.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        train = train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val = val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return train, val
    else:
        split =['test']
        ds = tfds.load(dataset, split=split,
                       shuffle_files=True,
                       as_supervised=True)[0]
        ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds


def fuse_conv_bn(conv, bn):
    kernel = conv.get_weights()[0]
    gamma = bn.gamma.numpy()
    beta = bn.beta.numpy()
    mean = bn.moving_mean.numpy()
    variance = bn.moving_variance.numpy()
    std = np.sqrt(variance + EPSILON)
    t = gamma / std
    t = t.reshape((1, 1, 1, -1))
    new_conv_kernel = (kernel * t)
    new_conv_bias = beta - (mean * gamma) / std
    return new_conv_kernel, new_conv_bias


def parameterize(org_model, new_model):
    # get / param weights for new model
    for layer in new_model.layers:
        layer_name = layer.name
        if 'parnet' in layer_name and '_conv3x3_conv' in layer_name:
            print(layer_name)
            # convert conv3x3 + BN to conv3x3
            org_conv3_layer = org_model.get_layer(name=layer_name)
            conv3_bn_name = layer_name.replace('_conv3x3_conv',
                                               '_conv3x3_bn')
            org_conv3_bn_layer = org_model.get_layer(name=conv3_bn_name)
            new_conv3_kernel, new_conv3_bias = fuse_conv_bn(
                org_conv3_layer,
                org_conv3_bn_layer
            )

            # convert conv1x1 + bn to conv1x1
            conv1_name = layer_name.replace('_conv3x3_conv',
                                            '_conv1x1_conv')
            conv1_bn_name = layer_name.replace('_conv3x3_conv',
                                               '_conv1x1_bn')
            org_conv1_layer = org_model.get_layer(name=conv1_name)
            org_conv1_bn_layer = org_model.get_layer(name=conv1_bn_name)
            new_conv1_kernel, new_conv1_bias = fuse_conv_bn(
                org_conv1_layer,
                org_conv1_bn_layer
            )

            # convert conv3x3 + conv1x1 to conv3x3
            new_bias = new_conv3_bias + new_conv1_bias
            pad_width = ((1, 1), (1, 1), (0, 0), (0, 0))
            padded_conv1_kernel = np.pad(new_conv1_kernel,
                                         pad_width,
                                         'constant',
                                         constant_values=(0, 0))
            new_kernel = new_conv3_kernel + padded_conv1_kernel

            # set weights
            layer.set_weights([new_kernel, new_bias])
        else:
            org_layer = org_model.get_layer(name=layer_name)
            layer.set_weights(org_layer.get_weights())

    return new_model
