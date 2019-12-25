import torch

from nlp.lang_translation.seq2seq.models import EncoderRNN, AttDecoderRNN


def test_encoder():
    batch_size = 50
    seq_len = 10
    hidden_units = 256
    # (batch_size, seq_len=1)
    X = torch.ones(batch_size, seq_len).long()

    vocab_size = 100
    encoder = EncoderRNN(vocab_size=vocab_size, batch_size=batch_size, hidden_units=hidden_units)
    output, hidden = encoder(X, encoder.default_hidden())
    print(output)
    print(hidden)


def test_decoder():
    batch_size = 50
    seq_len = 10
    hidden_units = 256
    vocab_size = 1000
    embedding_dim = 128

    decoder = DecoderRNN(embedding_dim=embedding_dim, vocab_size=vocab_size, hidden_units=hidden_units)

    X = torch.ones(batch_size, seq_len).long()
    h_0 = torch.zeros(1, batch_size, hidden_units)
    output, hidden = decoder(X, h_0)
    print(output.shape)
    print(hidden.shape)


def test_seq2seq():
    encoder_vocab_size = 1054
    decoder_vocab_size = 1200

    encoder_emb_dim = 120
    decoder_emb_dim = 200

    encoder_hidden_units = 127
    decoder_hidden_units = encoder_hidden_units

    batch_size = 54
    seq_len = 26

    encoder = EncoderRNN(vocab_size=encoder_vocab_size,
                         embedding_dim=encoder_emb_dim,
                         hidden_units=encoder_hidden_units,
                         batch_size=batch_size)
    encoder_input_tensor = torch.ones(batch_size, seq_len).long()
    encoder_h_0 = encoder.default_hidden()
    encoder_outputs, encoder_last_h_n = encoder(encoder_input_tensor, encoder_h_0)

    _decoder = AttDecoderRNN(vocab_size=decoder_vocab_size,
                             embedding_dim=decoder_emb_dim,
                             hidden_units=decoder_hidden_units,
                             max_encoder_seq_length=seq_len,
                             batch_size=batch_size)

    decoder_input_tensor = torch.ones(batch_size, 1).long()

    _decoder(decoder_input_tensor, encoder_last_h_n, encoder_outputs)

    print('bingo')


if __name__ == '__main__':
    # test_decoder()
    # test_encoder()
    test_seq2seq()
