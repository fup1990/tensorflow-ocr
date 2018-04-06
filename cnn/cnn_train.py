import time

from cnn import config as cfg
import numpy as np
import tensorflow as tf
from cnn import word_vec as wv

from cnn import gen_captcha as gc

slim = tf.contrib.slim

def variable_summary(name,var):
    with tf.name_scope("summaries"):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean/" + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev/" + name, stddev)

def next_batch(batch_size=64):
    # 图片数据
    batch_x = np.zeros((batch_size, cfg.IMAGE_WIDTH * cfg.IMAGE_HEIGHT))
    # 文字数据
    batch_y = np.zeros((batch_size, cfg.WORD_NUM * cfg.CHAR_NUM))

    for i in range(batch_size):
        text, image = gc.captcha_text_image(cfg.WORD_NUM)
        # 一维化
        batch_x[i, :] = image.reshape(-1) / 256
        batch_y[i, :] = wv.word2vec(text)

    return batch_x, batch_y

def inference(training=True, regularization=True):

    input_data = tf.placeholder(dtype=tf.float32, shape=[None, cfg.IMAGE_WIDTH * cfg.IMAGE_HEIGHT])
    label_data = tf.placeholder(dtype=tf.float32, shape=[None, cfg.WORD_NUM * cfg.CHAR_NUM])
    x = tf.reshape(input_data, shape=[-1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 1])

    with tf.variable_scope('conv1'):
        # 四维矩阵的权重参数，3, 3是过滤器的尺寸，1为图片深度， 64为filter数量
        weight1 = tf.get_variable('weights1', [3, 3, 1, 64], initializer=tf.random_normal_initializer(stddev=0.01))
        variable_summary('weights1', weight1)

        bias1 = tf.get_variable('bias1', [64], initializer=tf.constant_initializer(0.1))
        variable_summary('bias1', bias1)

        kernel1 = tf.nn.conv2d(x, weight1, strides=[1, 1, 1, 1], padding='SAME')
        # BN标准化
        bn1 = tf.contrib.layers.batch_norm(kernel1, is_training=True)
        conv1 = tf.nn.relu(tf.nn.bias_add(bn1, bias1))
        # conv1 = tf.nn.leaky_relu(tf.nn.bias_add(bn1, bias1))
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        lrn1 = tf.nn.lrn(pool1, name='lrn1')

    with tf.variable_scope('conv2'):
        weight2 = tf.get_variable('weights2', [3, 3, 64, 64], initializer=tf.random_normal_initializer(stddev=0.01))
        variable_summary('weights2', weight2)

        bias2 = tf.get_variable('bias2', [64], initializer=tf.constant_initializer(0.1))
        variable_summary('bias2', bias2)

        kernel2 = tf.nn.conv2d(lrn1, weight2, strides=[1, 1, 1, 1], padding='SAME')
        bn2 = tf.contrib.layers.batch_norm(kernel2, is_training=True)
        conv2 = tf.nn.relu(tf.nn.bias_add(bn2, bias2))
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        lrn2 = tf.nn.lrn(pool2, name='lrn2')

    with tf.variable_scope('conv3'):
        weight3 = tf.get_variable('weights3', [3, 3, 64, 128], initializer=tf.random_normal_initializer(stddev=0.1))
        variable_summary('weights3', weight3)

        bias3 = tf.get_variable('bias3', [128], initializer=tf.constant_initializer(0.1))
        variable_summary('bias3', bias3)

        kernel3 = tf.nn.conv2d(lrn2, weight3, strides=[1, 1, 1, 1], padding='SAME')
        bn3 = tf.contrib.layers.batch_norm(kernel3, is_training=True)
        conv3 = tf.nn.relu(tf.nn.bias_add(bn3, bias3))
        lrn3 = tf.nn.lrn(conv3, name='lrn2')
        pool3 = tf.nn.max_pool(lrn3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

    # 使用slim简写3层卷积层
    # conv = slim.repeat(x, 3, slim.conv2d, 64, [3, 3], scope='conv')
    # conv = slim.stack(x, slim.conv2d, [(64, [3, 3]), (64, [3, 3]), (128, [3, 3])], scope='conv')
    # pool = slim.max_pool2d(conv, [2, 2], scope='pool')

    # 计算将池化后的矩阵reshape成向量后的长度
    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 将池化后的矩阵reshape成向量
    dense = tf.reshape(pool3, [-1, nodes])

    with tf.variable_scope('fc1'):
        weight4 = tf.get_variable('weights4', [nodes, cfg.FULL_SIZE], initializer=tf.random_normal_initializer(stddev=0.01))
        variable_summary('weights4', weight4)

        bias4 = tf.get_variable('bias4', [cfg.FULL_SIZE], initializer=tf.constant_initializer(0.1))
        variable_summary('bias4', bias4)
        if regularization:
            tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(cfg.REGULARIZATION_RATE)(weight4))
        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense, weight4), bias=bias4))
        if training:
            fc1 = tf.nn.dropout(fc1, keep_prob=cfg.KEEP_PROB)

    with tf.variable_scope('fc2'):
        weight5 = tf.get_variable('weights5', [cfg.FULL_SIZE, cfg.WORD_NUM * cfg.CHAR_NUM], initializer=tf.random_normal_initializer(stddev=0.01))
        variable_summary('weights5', weight5)

        bias5 = tf.get_variable('bias5', [cfg.WORD_NUM * cfg.CHAR_NUM], initializer=tf.constant_initializer(0.1))
        variable_summary('bias5', bias5)

        if regularization:
            tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(cfg.REGULARIZATION_RATE)(weight5))
        outputs = tf.nn.bias_add(tf.matmul(fc1, weight5), bias5)
        # if training:
        #     outputs = tf.nn.dropout(outputs, keep_prob=cfg.KEEP_PROB)
    # fc2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(fc1, weight5), bias=bias5))


    # fc1 = slim.fully_connected(dense, FULL_SIZE, scope='fc1')
    # d1 = tf.nn.dropout(fc1, dropout)
    # fc2 = slim.fully_connected(d1, WORD_NUM * wv.CHAR_NUM, scope='fc2')
    # outputs = tf.nn.dropout(fc2, dropout)
    
    return input_data, label_data, outputs

def run_training():

    input_data, label_data, outputs = inference(training=True, regularization=True)
    with tf.variable_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_data, logits=outputs))
        tf.add_to_collection('loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('loss'))
        # global_step = tf.Variable(0)
        # learning_rate = tf.train.exponential_decay(cfg.LEARNING_RATE, global_step, 1000, 0.1)
        train_step = tf.train.AdamOptimizer(learning_rate=cfg.LEARNING_RATE).minimize(loss)
        variable_summary('loss', loss)

    with tf.variable_scope('accuracy'):
        max_idx_p = tf.argmax(tf.reshape(outputs, [-1, cfg.WORD_NUM, cfg.CHAR_NUM]), 2)
        max_idx_l = tf.argmax(tf.reshape(label_data, [-1, cfg.WORD_NUM, cfg.CHAR_NUM]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        variable_summary('accuracy', accuracy)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        checkpoint = tf.train.latest_checkpoint(cfg.CKPT_DIR)
        epoch = 0
        if checkpoint:
            saver.restore(sess, checkpoint)
            epoch += int(checkpoint.split('-')[-1])

        writer = tf.summary.FileWriter(cfg.LOG_DIR)
        try:
            while True:
                batch_x, batch_y = next_batch(128)
                _, l, summary_merged, acc = sess.run([train_step, loss, merged, accuracy], feed_dict={input_data: batch_x, label_data: batch_y})
                print('Epoch is {}, loss is {}, accuracy is {} time is {}'.format(epoch, l, acc, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                writer.add_summary(summary_merged)
                epoch += 1
                if epoch % 10 == 0:
                    saver.save(sess, cfg.CKPT_PATH, global_step=epoch)
        except Exception as e:
            print(e)
            saver.save(sess, cfg.CKPT_PATH, global_step=epoch)
            writer.close()

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()