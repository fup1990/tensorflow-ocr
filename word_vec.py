from gen_captcha import number, alphabet, ALPHABET
import numpy as np

char_set = number + alphabet + ALPHABET
char_num = len(number) + len(alphabet) + len(ALPHABET)

def word2vec(word):
    word_num = len(word)
    vec = np.zeros((word_num, char_num))
    for index, char in enumerate(word):
        i = char_set.index(char)
        vec[index][i] = 1
    return vec

def vec2word(vec):
    word = ''
    for i in range(len(vec)):
        char_vec = vec[i]
        no_zero = np.nonzero(char_vec)[0]
        for j in no_zero:
            word += char_set[j]
    return word