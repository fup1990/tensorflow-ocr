import tensorflow as tf
from cnn_train import inference, next_batch
from gen_captcha import captcha_text_image
from word_vec import vec2word
import numpy as np
import config as cfg

def predict_captcha():
    text, image = captcha_text_image(cfg.WORD_NUM)
    input_image = np.zeros((1, cfg.IMAGE_WIDTH * cfg.IMAGE_HEIGHT))
    input_image[0, :] = image.reshape(-1) / 256

    input_data, _, outputs = inference(training=False, regularization=False)
    # outputs = tf.nn.softmax(tf.reshape(outputs, [-1, cfg.WORD_NUM, cfg.CHAR_NUM]))
    # outputs = tf.reshape(outputs, [-1, cfg.WORD_NUM, cfg.CHAR_NUM])
    # prediction = tf.argmax(outputs, axis=2)
    prediction = tf.argmax(tf.reshape(outputs, [-1, cfg.WORD_NUM, cfg.CHAR_NUM]), 2)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        checkpoint = tf.train.latest_checkpoint(cfg.CKPT_PATH)
        if checkpoint:
            saver.restore(sess, checkpoint)

        vector = sess.run([prediction], feed_dict={input_data: input_image})
        vector = vector[0].tolist()
        output = np.zeros((cfg.WORD_NUM * cfg.CHAR_NUM))
        for i in range(cfg.WORD_NUM):
            output[i * cfg.CHAR_NUM + vector[i]] = 1
        predict_text = vec2word(output)
        print("正确: {}  预测: {}".format(text, predict_text))

def main(_):
    predict_captcha()

if __name__ == '__main__':
    tf.app.run()