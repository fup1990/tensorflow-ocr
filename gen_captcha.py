from captcha.image import ImageCaptcha
import random
from PIL import Image
import numpy as np
import config as cfg
import matplotlib.pyplot as plt

# 生成文字验证码
def random_captcha_text(char_set, size=4):
    text = []
    for i in range(size):
        r = random.choice(char_set)
        text.append(r)
    return text

def captcha_text_image(word_num):

    captcha_text = random_captcha_text(char_set=cfg.CHAR_SET, size=word_num)
    captcha_text = ''.join(captcha_text)

    # 导入验证码包 生成一张空白图
    image = ImageCaptcha(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, font_sizes=(25, 25, 56))
    captcha = image.generate(captcha_text)
    # 转换为图片格式
    captcha_image = Image.open(captcha)
    # 转化为numpy数组 shape=(60, 160, 3)
    captcha_image = np.array(captcha_image)

    captcha_image = convert2gray(captcha_image)

    return captcha_text, captcha_image

# 把彩色图像转为灰度图像
def convert2gray(img):
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

if __name__ == '__main__':
    text, image = captcha_text_image(cfg.WORD_NUM)
    print(text)
    plt.imshow(image)
    plt.show()