information = 'Using Simple-Seq2Seq Model ... ... '

enc_emb_dim = 128
enc_lstm_hidden_dim = 256

couplet_max_len = 15

dec_emb_dim = 128
# dec_lstm_hidden_dim = enc_lstm_hidden_dim * 2
dec_lstm_hidden_dim = 256

check_points_path = './resources/models/basic_seq2seq/checkpoints_basic_seq2seq.h5'
weight_model_path = './resources/models/basic_seq2seq/weight_basic_seq2seq.h5'
weight_encoder_path = './resources/models/basic_seq2seq/weight_basic_encoder.h5'
weight_decoder_path = './resources/models/basic_seq2seq/weight_basic_decoder.h5'
train_set_size = 76000
batch_size = 1000
steps_per_epoch = train_set_size / batch_size
epochs = 1000

PAD = 0
UNK = 1
GO = 2
EOS = 3

word_id_map_path = './resources/models/basic_seq2seq/word_id_map.h5'
id_word_map_path = './resources/models/basic_seq2seq/id_word_map.h5'

couplet_path = {
    'train': {'in': './resources/data/couplet/train/clean_in.txt',
              'out': './resources/data/couplet/train/clean_out.txt'},
    'dev': {'in': './resources/data/couplet/dev/clean_in.txt',
            'out': './resources/data/couplet/dev/clean_out.txt'},
    'test': {'in': './resources/data/couplet/test/clean_in.txt',
             'out': './resources/data/couplet/test/clean_out.txt'},
}

# 在已加载模型的情况下是否继续训练
insist_training = False
attention_dense_1_units = 10

model_image_path = './resources/images/couplet/simple_seq2seq.jpg'
