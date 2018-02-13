import tensorflow as tf
from cnn_train import WORD_NUM, cnn_outputs, X, dropout
from gen_captcha import captcha_text_image
from word_vec import vec2word, CHAR_NUM

def test_captcha():
    text, image = captcha_text_image(WORD_NUM)
    image = image.reshape(-1) / 256
    outputs = cnn_outputs()
    # prediction = tf.nn.softmax(outputs)
    outputs = tf.reshape(outputs, [-1, WORD_NUM, CHAR_NUM])
    prediction = tf.argmax(outputs, axis=2)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        vector = sess.run(prediction, feed_dict={X: [image], dropout: 1})
        predict_text = vec2word(vector)
        print("正确: {}  预测: {}".format(text, predict_text))

test_captcha()