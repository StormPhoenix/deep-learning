import jieba
import torch

import nlp.lang_translation.config as config
from nlp.lang_translation.data_utils import get_lang_dict
from nlp.lang_translation.seq2seq.models import EncoderRNN, AttDecoderRNN

device = torch.device('cpu')


def predict(input_tensor, words_len, encoder: EncoderRNN, decoder: AttDecoderRNN, target_dict):
    """

    :param input_tensor: shape = (batch_size, seq_len)
    :param words_len: list size = (batch_size)
    :param encoder:
    :param decoder:
    :return:
    """
    input_tensor.to(device=device)
    encoder.to(device=device)
    decoder.to(device=device)
    h_0 = encoder.default_hidden()
    h_0.to(device=device)

    encoder_outputs, hidden = encoder(input_tensor, words_len, h_0)

    # decoder_input shape = (batch_size, 1)
    decoder_input = torch.tensor([[target_dict.get_id('<SOS>')]], dtype=torch.long) \
        .repeat(input_tensor.shape[0], 1) \
        .to(device=device)

    # predict
    result = []
    decoder_hidden = hidden
    while True:
        # decoder_outputs shape = (batch_size, 1, vocab_size)
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # top_v shpae = (batch_size, 1, 1)
        top_v, top_i = decoder_output.topk(1, dim=-1)
        word_id = top_i.item()
        if word_id != target_dict.get_id('<EOS>') and len(result) < config.sequence_length:
            result.append(target_dict.get_word(word_id))
            decoder_input = torch.tensor([[word_id]], dtype=torch.long) \
                .repeat(input_tensor.shape[0], 1) \
                .to(device=device)
        else:
            result.append('<EOS>')
    return result


def parse_words_to_ids(words, ch_dict):
    """
    将句子转化成索引，同时做 padding
    :param words:
    :param ch_dict:
    :return:
    """
    words_ids = [ch_dict.get_id('<SOS>')]
    for word in words:
        words_ids.append(ch_dict.get_id(word))
    words_ids.append(ch_dict.get_id('<EOS>'))

    words_len = [len(words_ids)]

    while len(words_ids) < config.sequence_length:
        words_ids.append(ch_dict.get_id('<PAD>'))
    return words_ids, words_len


def main():
    ch_dict, en_dict = get_lang_dict()

    encoder: EncoderRNN = EncoderRNN(vocab_size=ch_dict.word_count,
                                     batch_size=config.batch_size,
                                     max_seq_len=config.sequence_length,
                                     hidden_units=config.hidden_units)
    # encoder.load_state_dict(torch.load(config.MODEL_ENCODER_PATH))
    # encoder.eval()

    decoder: AttDecoderRNN = AttDecoderRNN(vocab_size=en_dict.word_count,
                                           batch_size=config.batch_size,
                                           max_encoder_seq_length=config.sequence_length,
                                           hidden_units=config.hidden_units)
    # decoder.load_state_dict(torch.load(config.MODEL_DECODER_PATH))
    # decoder.eval()

    while True:
        sentence = input('输入句子:')
        words = list(jieba.cut(sentence, cut_all=True))
        word_ids, words_len = parse_words_to_ids(words, ch_dict)

        input_tensor = torch.tensor(word_ids, dtype=torch.long).unsqueeze(0)
        result = predict(input_tensor, words_len, encoder, decoder, en_dict)
        print(result)


if __name__ == '__main__':
    main()
