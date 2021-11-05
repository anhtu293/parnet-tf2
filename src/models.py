import tensorflow as tf
from src.layers import ParnetBlock, DownsamplingBlock, FusionBlock
import math


BASE_WIDTH = 64
CONFIG_BY_SCALE = {
    'small': {'scale': 1, 'final_down_filters': 1280},
    'medium': {'scale': 4/3, 'final_down_filters': 2048},
    'large': {'scale': 5/3, 'final_down_filters': 2560},
    'extra': {'scale': 25/12, 'final_down_filters': 3200}
}
IMAGENET_DEFAULT_BLOCKS_ARGS = {
    'common_head':[
        {'name': 'down_1', 'filters': 64, 'repeats': 1},
        {'name': 'down_2', 'filters': 96, 'repeats': 1},
        {'name': 'down_3', 'filters': 192, 'repeats': 1}],
    'stream_1': [
        {'name': 'stream1_parnet', 'filters': 96, 'repeats': 4},
        {'name': 'stream1_down_8', 'filters': 192, 'repeats': 1}],
    'stream_2':[
        {'name': 'stream2_parnet', 'filters': 192, 'repeats': 5},
        {'name': 'stream2_fusion_9', 'filters': 384, 'repeats': 1}],
    'stream_3':[
        {'name': 'down_4', 'filters': 384, 'repeats': 1},
        {'name': 'stream3_parnet', 'filters': 384, 'repeats': 5},
        {'name': 'stream3_fusion_10', 'filters': 384, 'repeats': 1}
    ]
}
CIFAR_DEFAULT_BLOCKS_ARGS = {
    'common_head':[
        {'name': 'head_parnet_1', 'filters': 64, 'repeats': 1},
        {'name': 'head_parnet_2', 'filters': 96, 'repeats': 1},
        {'name': 'down_3', 'filters': 192, 'repeats': 1}],
    'stream_1': [
        {'name': 'stream1_parnet', 'filters': 96, 'repeats': 3},
        {'name': 'stream1_down_8', 'filters': 192, 'repeats': 1}],
    'stream_2':[
        {'name': 'stream2_parnet', 'filters': 192, 'repeats': 4},
        {'name': 'stream2_fusion_9', 'filters': 384, 'repeats': 1}],
    'stream_3':[
        {'name': 'down_4', 'filters': 384, 'repeats': 1},
        {'name': 'stream3_parnet', 'filters': 384, 'repeats': 4},
        {'name': 'stream3_fusion_10', 'filters': 384, 'repeats': 1}
    ]
}


def ParNet(dataset='imagenet',
           scale='small',
           num_classes=1000,
           train=True,
           input_shape=(224, 224, 3)):
    assert dataset in ['imagenet', 'cifar10', 'cifar100'],\
            'Not supported dataset'
    assert scale in ['small', 'medium', 'large', 'extra'],\
        'Invalid scale'
    def round_filters(filters, scale):
        filters *= scale
        return int(math.ceil(filters))

    if dataset == 'imagenet':
        DEFAULT_BLOCKS_ARGS = IMAGENET_DEFAULT_BLOCKS_ARGS
    else:
        DEFAULT_BLOCKS_ARGS = CIFAR_DEFAULT_BLOCKS_ARGS

    config = CONFIG_BY_SCALE[scale]
    width_scale = config['scale']
    final_down_filters = config['final_down_filters']

    # input
    input = tf.keras.Input(shape=input_shape, name='input')
    
    # build head and streams
    streams = ['common_head', 'stream_1', 'stream_2', 'stream_3']
    stream_outputs = []
    head_outputs = []
    for stream in streams:
        # choose output
        if stream == 'common_head':
            outputs = head_outputs
        else:
            outputs = stream_outputs

        # choose input for stream
        if stream == 'common_head':
            s_input = input
        elif stream == 'stream_1':
            s_input = head_outputs[-2]
        else:
            s_input = head_outputs[-1]

        # get block args by stream
        block_args = DEFAULT_BLOCKS_ARGS[stream]
        for idx, arg in enumerate(block_args):
            for i in range(arg['repeats']):
                # get layer type
                if 'down' in arg['name']:
                    layer_fnc = DownsamplingBlock
                elif 'parnet' in arg['name']:
                    layer_fnc = ParnetBlock
                elif 'fusion' in arg['name']:
                    layer_fnc = FusionBlock

                # get args to create layer
                filters = round_filters(arg['filters'], width_scale)
                kwargs = {
                    'filters': filters,
                    'name': arg['name'] + '_{}'.format(i),
                    'train': train,
                    'groups': 2 if 'fusion' in arg['name'] else 1,
                    'strides': 2
                }
                if idx == 0 and i == 0:
                    output = layer_fnc(**kwargs)(s_input)
                else:
                    if 'fusion' not in arg['name']:
                        output = layer_fnc(**kwargs)(output)
                    else:
                        output = layer_fnc(**kwargs)(output, outputs[-1])
            if stream == 'common_head':
                outputs.append(output)
        if stream != 'common_head':
            outputs.append(output)
            print(outputs)

    # create last layers
    if dataset == 'imagenet':
        output = DownsamplingBlock(
            final_down_filters,
            name='final_down')(stream_outputs[-1])
    else:
        output = tf.keras.layers.Conv2D(
            final_down_filters,
            (1, 1),
            name='final_conv1x1')(stream_outputs[-1])
    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    output = tf.keras.layers.Dense(num_classes, name='fc')(output)

    # build model
    model = tf.keras.Model(inputs=input, outputs=output)
    return model
