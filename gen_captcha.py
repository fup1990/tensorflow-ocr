from captcha.image import ImageCaptcha
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 验证码中的字符
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# 生成文字验证码
def random_captcha_text(char_set=number + alphabet + ALPHABET, size=4):
    text = []
    for i in range(size):
        r = random.choice(char_set)
        text.append(r)
    return text

def captcha_text_image():

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    # 导入验证码包 生成一张空白图
    image = ImageCaptcha()
    captcha = image.generate(captcha_text)
    # 转换为图片格式
    captcha_image = Image.open(captcha)
    # 转化为numpy数组
    captcha_image = np.array(captcha_image)

    return captcha_text, captcha_image

captcha_text, captcha_image = captcha_text_image()
plt.imshow(captcha_image)
plt.show()