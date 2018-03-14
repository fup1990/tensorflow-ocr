import tensorflow as tf
slim = tf.contrib.slim
layers = tf.layers

is_training = True

x_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 32])

# CNN
with slim.arg_scope([slim.conv2d], stride=1, padding='VALID'):
    conv1 = slim.conv2d(x_data, 64, [3, 3], scope='conv1')
    pool2 = slim.max_pool2d(conv1, [2, 2], stride=2, scope='pool2')
    conv3 = slim.conv2d(pool2, 128, [3, 3], scope='conv3')
    pool4 = slim.max_pool2d(conv3, [2, 2], stride=2, scope='pool4')
    conv5 = slim.conv2d(pool4, 256, [3, 3], scope='conv5')
    conv6 = slim.conv2d(conv5, 256, [3, 3], scope='conv6')
    pool7 = slim.max_pool2d(conv6, [1, 2], stride=2, scope='pool7')
    conv8 = slim.conv2d(pool7, 512, [3, 3], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}, scope='conv8')
    conv10 = slim.conv2d(conv8, 512, [3, 3], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}, scope='conv10')
    pool12 = slim.max_pool2d(conv10, [1, 2], stride=2, scope='pool12')
    conv13 = slim.conv2d(pool12, 512, [2, 2], padding='SAME', scope='conv13')

# Map-to-Sequence

# RNN

# Transcription