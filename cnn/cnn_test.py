from cnn import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cnn import cnn_train as ct
from cnn import word_vec as wv

from cnn.gen_captcha import captcha_text_image


def predict_captcha():
    text, image = captcha_text_image(cfg.WORD_NUM)
    input_image = np.zeros((1, cfg.IMAGE_WIDTH * cfg.IMAGE_HEIGHT))
    input_image[0, :] = image.reshape(-1) / 256

    input_data, _, outputs = ct.inference(training=False, regularization=False)
    # outputs = tf.nn.softmax(tf.reshape(outputs, [-1, cfg.WORD_NUM, cfg.CHAR_NUM]))
    # outputs = tf.reshape(outputs, [-1, cfg.WORD_NUM, cfg.CHAR_NUM])
    # prediction = tf.argmax(outputs, axis=2)
    prediction = tf.argmax(tf.reshape(outputs, [-1, cfg.WORD_NUM, cfg.CHAR_NUM]), 2)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        checkpoint = tf.train.latest_checkpoint(cfg.CKPT_DIR)
        if checkpoint:
            saver.restore(sess, checkpoint)

        vector = sess.run([prediction], feed_dict={input_data: input_image})
        vector = vector[0].tolist()
        output = np.zeros((cfg.WORD_NUM * cfg.CHAR_NUM))
        i = 0
        for n in vector[0]:
            output[i * cfg.CHAR_NUM + n] = 1
            i += 1
        predict_text = wv.vec2word(output)
        print("正确: {}  预测: {}".format(text, predict_text))
        plt.imshow(image)
        plt.show()

def main(_):
    predict_captcha()

if __name__ == '__main__':
    tf.app.run()