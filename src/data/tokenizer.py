#!/usr/bin/env python3
# 2022 Ruchao Fan

import collections
import sentencepiece as spm


class SPTokenizer(object):
    def __init__(self, model_path, vocab_file):
        self.model = model_path
        self._build_sentence_piece_processor()
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_sentence_piece_processor(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model)

    def tokenize(self, text):
        split_tokens = self.sp.EncodeAsPieces(text)
        return split_tokens

    def text2tokens(self, text, addsos=False):
        """Converts a sequence of tokens into ids using the vocab."""
        tokens = self.tokenize(text)
        assert addsos == False
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def tokens2text(self, tokens):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        text = []
        for i in tokens:
            text.append(self.ids_to_tokens[i])
        return text


    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        vocab["blank"] = 0
        vocab["sos"] = 1
        vocab["eos"] = 2
        vocab["unk"] = 3
        index = 4
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

class CharTokenizer(object):
    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

    def tokenize(self, text):
        split_tokens = [x for x in text.strip()]
        return split_tokens

    def text2tokens(self, text, addsos=False):
        """Converts a sequence of tokens into ids using the vocab."""
        tokens = self.tokenize(text)
        assert addsos == False
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def tokens2text(self, tokens):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        text = []
        for i in tokens:
            text.append(self.ids_to_tokens[i])
        return text


    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        vocab["blank"] = 0
        vocab["sos"] = 1
        vocab["eos"] = 2
        vocab["unk"] = 3
        index = 4
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

