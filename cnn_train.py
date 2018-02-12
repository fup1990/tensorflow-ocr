import numpy as np
import gen_captcha as gc
import word_vec as wv
import tensorflow as tf

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
# 字符数量
WORD_NUM = 4

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

def train_cnn():
    X = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, WORD_NUM * wv.CHAR_NUM])
    dropout = tf.placeholder(dtype=tf.float32)

    # 将X转化为图片的shape(60，160，1)
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # 四维矩阵的权重参数，3, 3是过滤器的尺寸，1为图片深度， 32为过滤器深度
    w1 = tf.get_variable('weights1', [3, 3, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b1 = tf.get_variable('biases1', [32], initializer=tf.constant_initializer(0.1))
    c1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
    a1 = tf.nn.bias_add(c1, b1)
    actived1 = tf.nn.relu(a1)
    pool1 = tf.nn.max_pool(actived1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    d1 = tf.nn.dropout(pool1, dropout)

    w2 = tf.get_variable('weights2', [3, 3, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b2 = tf.get_variable('biases2', [64], initializer=tf.constant_initializer(0.1))
    c2 = tf.nn.conv2d(d1, w2, strides=[1, 1, 1, 1], padding='SAME')
    a2 = tf.nn.bias_add(c2, b2)
    actived2 = tf.nn.relu(a2)
    pool2 = tf.nn.max_pool(actived2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    d2 = tf.nn.dropout(pool2, dropout)

    w3 = tf.get_variable('weights3', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b3 = tf.get_variable('biases3', [64], initializer=tf.constant_initializer(0.1))
    c3 = tf.nn.conv2d(d2, w3, strides=[1, 1, 1, 1], padding='SAME')
    a3 = tf.nn.bias_add(c3, b3)
    actived3 = tf.nn.relu(a3)
    pool3 = tf.nn.max_pool(actived3, ksize=[1, 2, 2, 2], strides=[1, 1, 1, 1], padding='SAME')
    d3 = tf.nn.dropout(pool3, dropout)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        print(d3.shape)

train_cnn()