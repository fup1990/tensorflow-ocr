import numpy as np
import gen_captcha as gc
import word_vec as wv
import tensorflow as tf
import time
slim = tf.contrib.slim

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
# 字符数量
WORD_NUM = 4
# 全连接网络节点数量
FULL_SIZE = 512
# 持久化模型路径
CKPT_DIR = 'model/'
CKPT_PATH = CKPT_DIR + 'captcha.ckpt'

X = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT])
Y = tf.placeholder(dtype=tf.float32, shape=[None, WORD_NUM * wv.CHAR_NUM])
dropout = tf.placeholder(dtype=tf.float32)

def next_batch(batch_size=64):
    # 图片数据
    batch_x = np.zeros((batch_size, IMAGE_WIDTH * IMAGE_HEIGHT))
    # 文字数据
    batch_y = np.zeros((batch_size, WORD_NUM * wv.CHAR_NUM))

    for i in range(batch_size):
        text, image = gc.captcha_text_image(WORD_NUM)
        # 一维化
        image = image.reshape(-1) / 256
        batch_x[i, :] = image
        vec = wv.word2vec(text)
        batch_y[i, :] = vec.reshape(-1)

    return batch_x, batch_y

def cnn_outputs():

    # 将X转化为图片的shape(60，160，1)
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # 四维矩阵的权重参数，3, 3是过滤器的尺寸，1为图片深度， 64为filter数量
    weight1 = tf.get_variable('weights1', [3, 3, 1, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    bias1 = tf.get_variable('bias1', [64], initializer=tf.constant_initializer(0.1))
    kernel1 = tf.nn.conv2d(x, weight1, strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    # 输出shape(60, 160, 64)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # norm1 = tf.nn.lrn(pool1, )

    weight2 = tf.get_variable('weights2', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    bias2 = tf.get_variable('bias2', [64], initializer=tf.constant_initializer(0.1))
    kernel2 = tf.nn.conv2d(pool1, weight2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

    weight3 = tf.get_variable('weights3', [3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
    bias3 = tf.get_variable('bias3', [128], initializer=tf.constant_initializer(0.1))
    kernel3 = tf.nn.conv2d(pool2, weight3, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(tf.nn.bias_add(kernel3, bias3))
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

    # 使用slim简写3层卷积层
    # conv = slim.repeat(x, 3, slim.conv2d, 64, [3, 3], scope='conv')
    # conv = slim.stack(x, slim.conv2d, [(64, [3, 3]), (64, [3, 3]), (128, [3, 3])], scope='conv')
    # pool = slim.max_pool2d(conv, [2, 2], scope='pool')

    # 计算将池化后的矩阵reshape成向量后的长度
    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # 将池化后的矩阵reshape成向量
    dense = tf.reshape(pool3, [-1, nodes])

    weights4 = tf.get_variable('weights4', [nodes, FULL_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
    bias4 = tf.get_variable('bias4', [FULL_SIZE], initializer=tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense, weights4), bias=bias4))

    weight5 = tf.get_variable('weights5', [FULL_SIZE, WORD_NUM * wv.CHAR_NUM], initializer=tf.truncated_normal_initializer(stddev=0.1))
    bias5 = tf.get_variable('bias5', [WORD_NUM * wv.CHAR_NUM], initializer=tf.constant_initializer(0.1))
    outputs = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(local4, weight5), bias=bias5))


    # fc1 = slim.fully_connected(dense, FULL_SIZE, scope='fc1')
    # d1 = tf.nn.dropout(fc1, dropout)
    # fc2 = slim.fully_connected(d1, WORD_NUM * wv.CHAR_NUM, scope='fc2')
    # outputs = tf.nn.dropout(local5, dropout)
    
    return outputs

def run_training():
    outputs = cnn_outputs()

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=outputs))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        checkpoint = tf.train.latest_checkpoint(CKPT_DIR)
        epoch = 0
        if checkpoint:
            saver.restore(sess, checkpoint)
            epoch += int(checkpoint.split('-')[-1])

        while True:
            batch_x, batch_y = next_batch(64)
            _, accuracy = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, dropout: 0.75})
            print('Epoch is {}, loss is {}, time is {}'.format(epoch, accuracy, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            epoch += 1
            if epoch % 10 == 0:
                saver.save(sess, CKPT_PATH, global_step=epoch)

run_training()
