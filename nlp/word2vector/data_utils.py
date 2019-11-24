import jieba
import pandas as pd

NEG_COMMENT_XLS_PATH = './data/nlp/comment/neg.xls'
POS_COMMENT_XLS_PATH = './data/nlp/comment/pos.xls'


def load_comment_data():
    pos = pd.read_excel(NEG_COMMENT_XLS_PATH, header=None)
    pos['label'] = 1
    neg = pd.read_excel(POS_COMMENT_XLS_PATH, header=None)
    neg['label'] = 0
    all_ = pos.append(neg, ignore_index=False)
    all_['words'] = all_[0].apply(lambda s: list(jieba.cut(s)))

    words_list = list(all_['words'])
    return words_list


COUPLET_DEV_PATH = './data/nlp/couplet/dev/in.txt'


def load_couplet_data(limit=None):
    with open("./data/nlp/couplet/dev/in.txt", "r") as f:
        data_in = f.read()
    with open("./data/nlp/couplet/dev/out.txt", "r") as f:
        data_out = f.read()
    data_in_list = data_in.split("\n")
    data_out_list = data_out.split("\n")
    data_in_list = [data.split() for data in data_in_list]
    data_out_list = [data.split() for data in data_out_list]
    return data_in_list, data_out_list

# data = load_couplet_data()
# print('s')
