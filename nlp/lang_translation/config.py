# Data Set

# 翻译材料的最大长度，用来过滤太长的句子，实际长度需要加上2（<SOS>, <EOS>）
max_length = 50

sequence_length = max_length + 2

batch_size = 200

hidden_units = 128

MODEL_ENCODER_PATH = '../../resources/models/nlp/lang_translation/encoder.h5'
MODEL_DECODER_PATH = '../../resources/models/nlp/lang_translation/decoder.h5'

CHINESE_LANG_PATH = '../../resources/data/nlp/lang_translation/chinese_train'
ENGLISH_LANG_PATH = '../../resources/data/nlp/lang_translation/english_train'

LANG_DICT_PATH = '../../resources/data/nlp/lang_translation/{}.h5'
