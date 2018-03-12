import tensorflow as tf
slim = tf.contrib.slim
layers = tf.layers

x_data = tf.placeholder(dtype=tf.float32, shape=[None, 100, 32])

# CNN
conv1 = slim.conv2d(x_data, 64, [3, 3], stride=1, padding='VALID', scope='conv1')
pool2 = slim.max_pool2d(conv1, [2, 2], stride=2, scope='pool2')
conv3 = slim.conv2d(pool2, 128, [3, 3], stride=1, padding='VALID', scope='conv3')
pool4 = slim.max_pool2d(conv3, [2, 2], stride=2, scope='pool4')
conv5 = slim.conv2d(pool4, 256, [3, 3], stride=1, padding='VALID', scope='conv5')
conv6 = slim.conv2d(conv5, 256, [3, 3], stride=1, padding='VALID', scope='conv6')
pool7 = slim.max_pool2d(conv6, [1, 2], stride=2, scope='pool7')
conv8 = slim.conv2d(pool7, 512, [3, 3], stride=1, padding='VALID', scope='conv8')
bn9 = layers.batch_norm(conv8)
conv10 = slim.conv2d(bn9, 512, [3, 3], stride=1, padding='VALID', scope='conv10')
bn11 = layers.batch_norm(conv10)
pool12 = slim.max_pool2d(bn11, [1, 2], stride=2, scope='pool12')
conv13 = slim.conv2d(pool12, 512, [2, 2], stride=1, padding='SAME', scope='conv13')

# Map-to-Sequence

# RNN

# Transcription