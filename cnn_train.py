import numpy as np
import gen_captcha as gc
import word_vec as wv

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