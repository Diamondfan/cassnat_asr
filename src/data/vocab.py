#!/usr/bin/env python3
# 2020 Ruchao Fan

class Vocab(object):
    def __init__(self, vocab_file, rank):
        self.vocab_file = vocab_file
        self.word2index = {"blank": 0, "sos": 1, "eos": 2, "unk": 3}
        self.index2word = {0: "blank", 1: "sos", 2: "eos", 3: "unk"}
        self.word2count = {}
        self.n_words = 4
        self.rank = rank
        self.read_lang()

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def read_lang(self):
        if self.rank == 0:
            print("Reading vocabulary from {}".format(self.vocab_file))
        with open(self.vocab_file, 'r') as rf:
            line = rf.readline()
            while line:
                line = line.strip().split(' ')
                if len(line) > 1:
                    sentence = ' '.join(line[1:])
                else:
                    sentence = line[0]
                self.add_sentence(sentence)
                line = rf.readline()
        if self.rank == 0:
            print("Vocabulary size is {}".format(self.n_words))


