import logging
import os

from gensim.models import fasttext

from nlp.word2vector.data_utils import load_comment_data, load_couplet_data

COMMENT_FASTTEXT_MODEL_PATH = './nlp/models/fasttext_comment.models'
COUPLET_FASTTEXT_MODEL_PATH = './nlp/models/fasttext_couplet.models'


def load_fasttext_comment_model():
    if os.path.exists(COMMENT_FASTTEXT_MODEL_PATH):
        model = fasttext.FastText.load(COMMENT_FASTTEXT_MODEL_PATH)
    else:
        words_list = load_comment_data()
        model = fasttext.FastText(words_list, min_count=1, iter=20)
        model.save(COMMENT_FASTTEXT_MODEL_PATH)
    return model


def load_fasttext_couplet_model():
    if os.path.exists(COUPLET_FASTTEXT_MODEL_PATH):
        model = fasttext.FastText.load(COUPLET_FASTTEXT_MODEL_PATH)
    else:
        words_list = load_couplet_data()
        model = fasttext.FastText(words_list, min_count=1, iter=20)
        model.save(COUPLET_FASTTEXT_MODEL_PATH)
    return model


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = load_fasttext_couplet_model()
    print(model.wv.most_similar('夜'))
