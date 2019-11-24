import logging
import os

from gensim.models import word2vec

from nlp.word2vector.data_utils import load_comment_data, load_couplet_data

COMMENT_W2V_MODEL_PATH = './nlp/models/word2vec_comment.models'
COUPLET_W2V_MODEL_PATH = './nlp/models/word2vec_couplet.models'


def load_word2vector_comment_model():
    if os.path.exists(COMMENT_W2V_MODEL_PATH):
        model = word2vec.Word2Vec.load(COMMENT_W2V_MODEL_PATH)
    else:
        words_list = load_comment_data()
        model = word2vec.Word2Vec(words_list, min_count=1, iter=20)
        model.save(COMMENT_W2V_MODEL_PATH)


def load_word2vector_couplet_model():
    if os.path.exists(COUPLET_W2V_MODEL_PATH):
        model = word2vec.Word2Vec.load(COUPLET_W2V_MODEL_PATH)
    else:
        words_list = load_couplet_data()
        model = word2vec.Word2Vec(words_list, min_count=1, iter=20)
        model.save(COUPLET_W2V_MODEL_PATH)
    return model


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = load_word2vector_couplet_model()

    print(model.wv.most_similar('夜'))
