import tensorflow as tf
from tensorflow.keras import layers, activations


class Shuffle(layers.Layer):
    def __init__(self, **kwargs):
        super(Shuffle, self).__init__(**kwargs)

    def call(self, x):
        c_idx = tf.range(0, tf.shape(x)[-1])
        c_idx = tf.random.shuffle(c_idx)
        x = tf.gather(x, c_idx, axis=-1)
        return x


class Silu(layers.Layer):
    def __init__(self, **kwargs):
        super(Silu, self).__init__(**kwargs)
        self.activation = tf.nn.silu

    def call(self, inputs):
        return self.activation(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'activation': activations.serialize(self.activation)}
        base_config = super(Silu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SSEBlock(filters, name, **kargs):
    def wrapper(x):
        # if input and output of ParNet block have
        # different number of channels (cifar), we need to pass input
        # through conv1x1 to fit number of channels
        if x.shape[-1] != filters:
            x = layers.Conv2D(filters,
                              (1, 1),
                              use_bias=False,
                              name=name + '_conv1x1')(x)
        x = layers.BatchNormalization(name=name + '_bn')(x)
        se = layers.GlobalAveragePooling2D(
            keepdims=True, name=name + '_se_gp')(x)
        se = layers.Conv2D(filters,
                           (1, 1),
                           use_bias=False,
                           name=name + '_se_conv1x1')(se)
        se = layers.Activation('sigmoid', name=name + '_se_sigmoid')(se)
        output = layers.Multiply(name=name + '_multiply')([x, se])
        return output
    return wrapper


def ParnetBlock(filters, name, train=True, **kwargs):
    def wrapper(x):
        se = SSEBlock(filters, name + '_sse')(x)
        conv3 = layers.Conv2D(filters,
                              (3, 3),
                              padding='same',
                              use_bias=not train,
                              name=name + '_conv3x3_conv')(x)
        if train:
            conv3 = layers.BatchNormalization(name=name + '_conv3x3_bn')(conv3)
            conv1 = layers.Conv2D(filters,
                                  (1, 1),
                                  use_bias=False,
                                  name=name + '_conv1x1_conv')(x)
            conv1 = layers.BatchNormalization(name=name + '_conv1x1_bn')(conv1)
            output = layers.Add(name=name + '_add')([se, conv1, conv3])
        else:
            output = layers.Add(name=name + '_add')([se, conv3])
        output = Silu(name=name + '_silu')(output)
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
                               use_bias=False,
                               name=name + '_pool2d_conv1')(pool_1)
        pool_1 = layers.BatchNormalization(name=name + '_pool2d_bn')(pool_1)

        conv = layers.Conv2D(filters,
                             (3, 3),
                             strides=strides,
                             padding='same',
                             groups=groups,
                             use_bias=False,
                             name=name + '_conv3_conv')(x)
        conv = layers.BatchNormalization(name=name + '_conv3_bn')(conv)

        global_pool = layers.GlobalAveragePooling2D(
            keepdims=True,
            name=name + '_gp_pool')(x)
        global_pool = layers.Conv2D(filters,
                                    (1, 1),
                                    strides=strides,
                                    groups=groups,
                                    use_bias=False,
                                    name=name + '_gp_conv')(global_pool)
        global_pool = layers.Activation('sigmoid',
                                        name=name + '_sigmoid')(global_pool)

        output = layers.Add(name=name + '_add')([pool_1, conv])
        output = layers.Multiply(name=name + '_multiply')([output, global_pool])
        output = Silu(name=name + '_silu')(output)
        return output
    return wrapper


def FusionBlock(filters, name, strides=2, groups=2, **kwargs):
    def wrapper(x, y):
        merged = layers.Concatenate(axis=-1,
                                    name=name + '_concatenation')([x, y])
        # merged = Shuffle(name=name + '_shuffle')(merged)
        output = DownsamplingBlock(filters,
                                   name + '_downsampling',
                                   strides,
                                   groups)(merged)
        return output
    return wrapper
