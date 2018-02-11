from gen_captcha import number, alphabet, ALPHABET
import numpy as np

CHAR_SET = number + alphabet + ALPHABET
CHAR_NUM = len(number) + len(alphabet) + len(ALPHABET)

def word2vec(word):
    word_num = len(word)
    vec = np.zeros((word_num, CHAR_NUM))
    for index, char in enumerate(word):
        i = CHAR_SET.index(char)
        vec[index][i] = 1
    return vec

def vec2word(vec):
    word = ''
    for i in range(len(vec)):
        char_vec = vec[i]
        no_zero = np.nonzero(char_vec)[0]
        for j in no_zero:
            word += CHAR_SET[j]
    return word
