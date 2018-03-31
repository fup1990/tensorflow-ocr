import tensorflow as tf
import numpy as np

def conv2d(input_data, out_channel, ksize=3, strides=1, padding=1, w_init=None, b_init=None, name=None):

    in_shape = input_data.get_shape().as_list()

    if padding == 1:
        padding = 'VALID'
    else:
        padding = 'SAME'

    if isinstance(ksize, list):
        # in_shape[3]获取图片的深度
        filter = ksize + [in_shape[3], out_channel]
    else:
        filter = [ksize, ksize, in_shape[3], out_channel]

    if isinstance(strides, list):
        strides = [1, strides[0], strides[1], 1]
    else:
        strides = [1, strides, strides, 1]

    if w_init is None:
        # he initial
        w_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()

    weights = tf.get_variable("weights", filter, initializer=w_init)
    bias = tf.get_variable('bias', [out_channel], initializer=b_init)

    conv = tf.nn.conv2d(input_data, weights, strides, padding, name=name)

    return relu(tf.nn.bias_add(conv, bias=bias))

def max_pooling(value, ksize=2, strides=2, padding=0, data_format="NHWC", name=None):

    if padding == 1:
        padding = 'VALID'
    else:
        padding = 'SAME'

    if isinstance(ksize, list):
        ksize = [1, ksize[0], ksize[1], 1]
    else:
        ksize = [1, ksize, ksize, 1]

    if strides is None:
        strides = ksize

    if isinstance(strides, list):
        strides = [1, strides[0], strides[1], 1]
    else:
        strides = [1, strides, strides, 1]

    return tf.nn.max_pool(value, ksize, strides, padding, data_format=data_format, name=name)

def relu(features, name=None):
    return tf.nn.relu(features, name=name)

def batch_norm(input_data):
    return tf.contrib.layers.batch_norm(input_data, is_training=True)

# CNN
def conv(input_data):
    """
    :param input_data:batch*32*100*3
    :return: output_data:batch*1*25*512
    """
    conv1 = conv2d(input_data, out_channel=64)
    pool2 = max_pooling(conv1)                                          # batch*16*50*64
    conv3 = conv2d(pool2, out_channel=128)
    pool4 = max_pooling(conv3)                                          # batch*8*25*128
    conv5 = conv2d(pool4, out_channel=256)
    conv6 = conv2d(conv5, out_channel=256)
    pool7 = max_pooling(conv6, ksize=[2, 1], strides=[2, 1])            # batch*4*25*256
    conv8 = conv2d(pool7, out_channel=512)
    bn9 = batch_norm(conv8)
    conv10 = conv2d(bn9, out_channel=512)
    bn11 = batch_norm(conv10)
    pool12 = max_pooling(bn11, ksize=[2, 1], strides=[2, 1])            # batch*2*25*512
    conv13 = conv2d(pool12, out_channel=512, ksize=2, strides=[2, 1], padding=0)
    return conv13

# Map-to-Sequence
def map_to_sequence(input_data):
    """
    :param input_data:batch*1*25*512
    :return:output_data:batch*25*512
    """
    shape = input_data.get_shape().as_list()
    assert shape[1] == 1
    # 从tensor中删除所有大小是1的维度
    return tf.squeeze(input_data)

# BiRNN
def birnn():


# Transcription

