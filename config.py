# 图像大小
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 80
# 字符数量
WORD_NUM = 4
# 全连接网络节点数量
FULL_SIZE = 1024
# 持久化模型路径
CKPT_DIR = 'model/'
CKPT_PATH = CKPT_DIR + 'captcha.ckpt'
# tensorbord日志路径
LOG_DIR = 'log/'
# 正则化速率
REGULARIZATION_RATE = 0.001
# 验证码中的字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
CHAR_SET = number + alphabet + ALPHABET
CHAR_NUM = len(number) + len(alphabet) + len(ALPHABET)
# dropout
KEEP_PROB = 0.75
LEARNING_RATE = 0.001
