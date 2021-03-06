import random

import torch
import torch.nn as nn
import torch.optim as optimizer
from tqdm import tqdm

import nlp.lang_translation.config as config
from nlp.lang_translation.data_utils import get_text_pairs
from nlp.lang_translation.seq2seq.models import EncoderRNN, AttDecoderRNN, device

teacher_forceing_ratio = 0.8
learning_rate = 0.01

SOS_Token = 1


def get_loss(input_batch, input_length_batch, target_batch, encoder: EncoderRNN, decoder: AttDecoderRNN,
             encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    h_0 = encoder.default_hidden()
    encoder_outputs, hidden = encoder(input_batch, input_length_batch, h_0)

    # all teaching
    decoder_input = torch.Tensor([[SOS_Token]]).repeat(config.batch_size, 1).long().to(device)

    loss = 0
    target_length = target_batch.shape[1]
    decoder_hidden = h_0

    use_teacher_forcing = True if random.random() < teacher_forceing_ratio else False
    if use_teacher_forcing:
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output.squeeze(1), target_batch[:, i])
            decoder_input = target_batch[:, i].unsqueeze(-1)
    else:
        for i in range(target_length):
            # decoder_output shape = (batch_size, 1, vocab_size)
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # top_i shape = [batch_size, 1, 1]
            top_v, top_i = decoder_output.topk(1, dim=-1)
            decoder_input = top_i.squeeze(1).detach()

            loss += criterion(decoder_output.squeeze(1), target_batch[:, i])

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def pad_sequence(translation_pairs, input_lang_dict, target_lang_dict):
    input_lang = translation_pairs[0]
    target_lang = translation_pairs[1]

    if len(input_lang) != len(target_lang):
        print('lang pair length is not equal')
        return

    length = len(input_lang)

    input_tensor = []
    input_tensor_length = []
    target_tensor = []

    # 为了防止出现 bug，这里手动将 input_lang 的第一条数据手动做 padding ， 然后再调用 pad_sequence
    while len(input_lang[0]) < config.sequence_length:
        input_lang[0].append(input_lang_dict.get_id('<PAD>'))

    for i in range(length):
        input_tensor.append(torch.tensor(input_lang[i], dtype=torch.long, device=device))
        input_tensor_length.append(len(input_lang[i]))
        target_tensor.append(torch.tensor(input_lang[i], dtype=torch.long, device=device))

    input_tensor = nn.utils.rnn.pad_sequence(input_tensor, batch_first=True,
                                             padding_value=input_lang_dict.get_id('<PAD>'))
    target_tensor = nn.utils.rnn.pad_sequence(target_tensor, batch_first=True,
                                              padding_value=target_lang_dict.get_id('<PAD>'))

    return input_tensor, input_tensor_length, target_tensor


def train(input_tensor, input_tensor_length, target_tensor, input_lang_dict, output_lang_dict, encoder: EncoderRNN,
          decoder, epochs):
    # 填充 translation_pairs

    encoder_optimizer = optimizer.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optimizer.SGD(decoder.parameters(), lr=learning_rate)

    # reference : https://blog.csdn.net/qq_22210253/article/details/85229988
    criterion = nn.NLLLoss()

    dataset_size = input_tensor.shape[0]
    epoch_count = int(dataset_size / config.batch_size)

    for epoch in range(epochs):
        loss = 0
        print('epoch : {}'.format(epoch))
        for e in tqdm(range(epoch_count)):
            start_index = e * config.batch_size
            end_index = start_index + config.batch_size
            input_tensor_batch = input_tensor[start_index: end_index]
            input_tensor_length_batch = input_tensor_length[start_index: end_index]
            target_tensor_batch = target_tensor[start_index: end_index]

            loss += get_loss(input_tensor_batch,
                             input_tensor_length_batch,
                             target_tensor_batch,
                             encoder, decoder,
                             encoder_optimizer,
                             decoder_optimizer,
                             criterion)

        torch.save(encoder.state_dict(), config.MODEL_ENCODER_PATH)
        torch.save(decoder.state_dict(), config.MODEL_DECODER_PATH)

        print('total loss : {}'.format(loss))

    print('bingo')


def main():
    # 得到的 translation_pairs 的翻译对中，每一句话都加上了 <SOS> 和 <EOS>
    translation_pairs, ch_lang_dict, en_lang_dict = get_text_pairs()

    input_tensor, input_tensor_length, target_tensor = pad_sequence(translation_pairs, ch_lang_dict, en_lang_dict)

    in_vocab_size = ch_lang_dict.word_count
    out_vocab_size = en_lang_dict.word_count

    encoder = EncoderRNN(vocab_size=in_vocab_size,
                         batch_size=config.batch_size,
                         max_seq_len=config.sequence_length,
                         hidden_units=config.hidden_units).to(device)
    decoder = AttDecoderRNN(vocab_size=out_vocab_size,
                            max_encoder_seq_length=config.sequence_length,
                            batch_size=config.batch_size,
                            hidden_units=config.hidden_units).to(device)
    train(input_tensor=input_tensor,
          input_tensor_length=input_tensor_length,
          target_tensor=target_tensor,
          input_lang_dict=ch_lang_dict,
          output_lang_dict=en_lang_dict,
          encoder=encoder,
          decoder=decoder, epochs=500)

    print('bingo')


if __name__ == '__main__':
    main()
