import tensorflow as tf
from cnn_train import WORD_NUM, cnn_outputs, X, dropout, CKPT_DIR
from gen_captcha import captcha_text_image
from word_vec import vec2word, CHAR_NUM
import numpy as np

def test_captcha():
    text, image = captcha_text_image(WORD_NUM)
    image = image.reshape(-1) / 256
    outputs = cnn_outputs()
    outputs = tf.reshape(outputs, [-1, WORD_NUM, CHAR_NUM])
    prediction = tf.nn.softmax(outputs)
    prediction = tf.argmax(prediction[0], axis=1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        checkpoint = tf.train.latest_checkpoint(CKPT_DIR)
        if checkpoint:
            saver.restore(sess, checkpoint)

        vector = sess.run(prediction, feed_dict={X: [image], dropout: 1})

        output = np.zeros((WORD_NUM, CHAR_NUM))
        for i in range(len(vector)):
            index = vector[i]
            output[i][index] = 1
        predict_text = vec2word(output)
        print("正确: {}  预测: {}".format(text, predict_text))
