import numpy as np
import config as cfg


def word2vec(word):
    word_num = len(word)
    vec = np.zeros((word_num, cfg.CHAR_NUM))
    for index, char in enumerate(word):
        i = cfg.CHAR_SET.index(char)
        vec[index][i] = 1
    return vec

def vec2word(vec):
    word = ''
    for i in range(len(vec)):
        char_vec = vec[i]
        no_zero = np.nonzero(char_vec)[0]
        for j in no_zero:
            word += cfg.CHAR_SET[j]
    return word

def main():
    vec = word2vec('12ab')
    print(vec)
    word = vec2word(vec)
    print(word)

if __name__ == '__main__':
    main()