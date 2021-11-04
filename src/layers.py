import tensorflow as tf
from tensorflow.keras import layers, activations


def SSEBlock(filters, name, **kargs):
    def wrapper(x):
        # if input and output of ParNet block have
        # different number of channels (cifar), we need to pass input
        # through conv1x1 to fit number of channels
        if x.shape[-1] != filters:
            x = layers.Conv2D(filters,
                              (1, 1),
                              name=name + '_conv1x1')(x)
        x = layers.BatchNormalization(name=name + '_bn')(x)
        se = layers.GlobalAveragePooling2D(
            keepdims=True, name=name + '_se_gp')(x)
        se = layers.Conv2D(filters,
                           (1, 1),
                           name=name + '_se_conv1x1')(se)
        se = activations.sigmoid(se, name=name + '_se_sigmoid')
        output = tf.math.multiply(x, se, name=name + '_multiply')
        return output
    return wrapper


def ParnetBlock(filters, name, train=True, **kwargs):
    def wrapper(x):
        se = SSEBlock(filters, name + '_sse')(x)
        conv3 = layers.Conv2D(filters,
                              (3, 3),
                              padding='same',
                              name=name + '_conv3_conv')(x)
        conv3 = layers.BatchNormalization(name=name + '_conv3x3_bn')(conv3)
        if train:
            conv1 = layers.Conv2D(filters,
                                  (1, 1),
                                  name=name + '_conv1_conv')(x)
            conv1 = layers.BatchNormalization(name=name + '_conv1_bn')(conv1)
            output = tf.math.add_n([se, conv1, conv3], name=name + '_add')
        else:
            output = tf.math.add_n([se, conv3], name=name + '_add')
        output = tf.nn.silu(output, name=name + '_silu')
        return output
    return wrapper


def DownsamplingBlock(filters, name, strides=2, groups=1, **kwargs):
    def wrapper(x):
        pool_1 = layers.AveragePooling2D((2, 2),
                                         padding='same',
                                         name=name + '_pool2d_pool')(x)
        pool_1 = layers.Conv2D(filters,
                               (1, 1),
                               groups=groups,
                               name=name + '_pool2d_conv1')(pool_1)
        pool_1 = layers.BatchNormalization(name=name + '_pool2d_bn')(pool_1)

        conv = layers.Conv2D(filters,
                             (3, 3),
                             strides=2,
                             padding='same',
                             groups=groups,
                             name=name + '_conv3_conv')(x)
        conv = layers.BatchNormalization(name=name + '_conv3_bn')(conv)

        global_pool = layers.GlobalAveragePooling2D(
            keepdims=True,
            name=name + '_gp_pool')(x)
        global_pool = layers.Conv2D(filters,
                                    (1, 1),
                                    strides=2,
                                    groups=groups,
                                    name=name + '_gp_conv')(global_pool)
        global_pool = activations.sigmoid(global_pool, name=name + '_sigmoid')

        output = tf.math.add(pool_1, conv, name=name + '_add')
        output = tf.math.multiply(output,
                                  global_pool,
                                  name=name + '_multiply')
        output = tf.nn.silu(output, name=name + '_silu')
        return output
    return wrapper


def FusionBlock(filters, name, strides=2, groups=2, **kwargs):
    def wrapper(x, y):
        merged = layers.Concatenate(axis=-1,
                                    name=name + '_concatenation')([x, y])
        c_idx = tf.range(0, tf.shape(merged)[-1])
        c_idx = tf.random.shuffle(c_idx)
        merged = tf.gather(merged, c_idx)
        # reassign shape after gather
        output = DownsamplingBlock(filters,
                                   name + '_downsampling',
                                   strides,
                                   groups)(merged)
        return output
    return wrapper
