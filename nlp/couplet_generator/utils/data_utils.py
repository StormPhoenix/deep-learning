import os
import pickle

import pandas as pd

import nlp.couplet_generator.config as config


def build_vocab():
    if os.path.exists(config.word_id_map_path) and os.path.exists(config.id_word_map_path):
        with open(config.word_id_map_path, 'rb') as fr:
            word_id_map = pickle.load(fr)

        with open(config.id_word_map_path, 'rb') as fr:
            id_word_map = pickle.load(fr)
    else:
        couplet_in, couplet_out = load_clean_couplet_data(couplet_in_path=config.couplet_path['dev']['in'],
                                                          couplet_out_path=config.couplet_path['dev']['out'])
        all_ = couplet_in.append(couplet_out, ignore_index=True)
        content = ''.join(all_['couplets'])
        word_set = set(list(content))

        word_id_map = {word: index + 4 for index, word in enumerate(word_set)}
        word_id_map['<pad>'] = 0
        word_id_map['<unk>>'] = 1
        word_id_map['<go>'] = 2
        word_id_map['<eos>'] = 3

        id_word_map = {index: word for word, index in word_id_map.items()}

        with open(config.word_id_map_path, 'wb') as fr:
            pickle.dump(word_id_map, fr)

        with open(config.id_word_map_path, 'wb') as fr:
            pickle.dump(id_word_map, fr)

        del content
        del couplet_in
        del couplet_out
        del all_
    return word_id_map, id_word_map


def couplets_generator(train_set_size=20000,
                       couplet_in_path='./resources/data/couplet/dev/clean_in.txt',
                       couplet_out_path='./resources/data/couplet/dev/clean_out.txt'):
    while True:
        fr_in = open(couplet_in_path, "r", encoding='utf8')
        fr_out = open(couplet_out_path, "r", encoding='utf8')

        line_in = fr_in.readline()
        line_out = fr_out.readline()
        count = 0
        while not empty(line_in) and not empty(line_out):
            if line_in.find('couplets') is not -1 or line_out.find('couplets') is not -1:
                line_in = fr_in.readline()
                line_out = fr_out.readline()
                continue
            count += 1
            line_in = (line_in[:-1])
            line_out = (line_out[:-1])
            yield line_in, line_out
            if count > train_set_size:
                break
            line_in = fr_in.readline()
            line_out = fr_out.readline()
        fr_in.close()
        fr_out.close()


def empty(input: str):
    return input is None or len(input) is 0 or input.isspace()


def load_clean_couplet_data(couplet_in_path='./resources/data/couplet/dev/clean_in.txt',
                            couplet_out_path='./resources/data/couplet/dev/clean_out.txt'):
    couplet_in = pd.read_csv(couplet_in_path, encoding='utf8')
    couplet_out = pd.read_csv(couplet_out_path, encoding='utf8')
    return couplet_in, couplet_out


def clean_couplet_data(couplet_in_path='./resources/data/couplet/dev/clean_in.txt',
                       couplet_out_path='./resources/data/couplet/dev/clean_out.txt'):
    couplet_in = pd.read_table(couplet_in_path, encoding='utf8')
    couplet_out = pd.read_table(couplet_out_path, encoding='utf8')

    def clean(sen: str):
        sen = sen.strip('\n')
        return ''.join(sen.split())

    couplet_in['couplets'] = couplet_in['couplets'].apply(lambda x: clean(x))
    couplet_out['couplets'] = couplet_out['couplets'].apply(lambda x: clean(x))

    couplet_in['len'] = couplet_in['couplets'].apply(lambda x: len(x))
    couplet_out['len'] = couplet_out['couplets'].apply(lambda x: len(x))

    couplet_in = couplet_in.loc[couplet_in['len'] > 4]
    couplet_out = couplet_out.loc[couplet_out['len'] > 4]

    couplet_in = couplet_in.drop(['len'], axis=1)
    couplet_out = couplet_out.drop(['len'], axis=1)

    couplet_in.to_csv(couplet_in_path, index=None, header=None, encoding='utf8')
    couplet_out.to_csv(couplet_out_path, index=None, header=None, encoding='utf8')


if __name__ == '__main__':
    clean_couplet_data()
    # clean_couplet_data()
    # gen = cleaned_couplets_generator()
    # while True:
    #     result = next(gen)
    #     print(result)
    # print('couplet generator')
