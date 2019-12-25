import os
import pickle

import nlp.lang_translation.config as config


class Lang:
    def __init__(self, name):
        self.lang_name = 'lang_{}'.format(name)
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_count = 4

    def get_id(self, word):
        return self.word2index.get(word, 3)

    def get_word(self, id):
        return self.index2word.get(id, '<UNK>')

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word=word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.word_count
            self.index2word[self.word_count] = word
            self.word_count += 1

    def parse_words_to_ids(self, words):
        parsed_words = [self.word2index.get('<SOS>')]
        if words is not None:
            for word in words:
                word_index = self.word2index.get(word, 2)
                if word_index != 2:
                    parsed_words.append(word_index)
                else:
                    # Unknown
                    self.add_word(word)
                    word_index = self.word2index.get(word)
                    parsed_words.append(word_index)
        parsed_words.append(self.word2index.get('<EOS>'))
        return parsed_words

    def is_dict_exists(self):
        return os.path.exists(config.LANG_DICT_PATH.format(self.lang_name))

    def load(self):
        with open(config.LANG_DICT_PATH.format(self.lang_name), 'rb') as fr:
            lang_dict = pickle.load(fr)
            self.word2index = lang_dict['word2index']
            self.index2word = lang_dict['index2word']
            self.word_count = lang_dict['word_count']

    def save(self):
        lang_dict = {
            'word2index': self.word2index,
            'index2word': self.index2word,
            'word_count': self.word_count
        }

        with open(config.LANG_DICT_PATH.format(self.lang_name), 'wb') as fw:
            pickle.dump(lang_dict, fw)
