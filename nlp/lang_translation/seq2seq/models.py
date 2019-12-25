'''
GRU
constructro parameters:
    input_size
    hidden_size
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_units=128, batch_size=64, max_seq_len=20):
        """
        encoder 初始化
        :param vocab_size: 词典大小
        :param embedding_dim: 单词编码维度
        :param hidden_size: encoder 隐藏层大小
        :param num_layers: gru 层数
        :param batch_size: 每批次数据大小
        """
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.bidirectional = False
        self.hidden_size = hidden_units
        self.num_layers = 1
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_units,
                          num_layers=self.num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=True)

    def forward(self, x, x_length, h_0):
        """
        输入 x、 h_0 到 gru，返回的 output、 hidden 的 shape 为
        output.shape : (batch_size, seq_len, self.hidden_units)
        hidden.shape : (1, batch_size, self.hidden_units)

        :param x:
        :param h_0:
        :return: gru 的结果
        """
        # embedded.shape = (batch_size, seq_len, embedded_dim)
        embedded = self.embedding(x)

        pack_embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                          x_length,
                                                          batch_first=True,
                                                          enforce_sorted=False)

        # output, hidden = self.gru(embedded, h_0)
        output, hidden = self.gru(pack_embedded, h_0)
        output, output_len = nn.utils.rnn.pad_packed_sequence(output, total_length=self.max_seq_len, batch_first=True)
        return output, hidden

    def default_hidden(self):
        # shape = (num_layers * num_directions, batch_size, hidden_size) batch_size 的位置和 batch_first 参数无关
        if self.bidirectional:
            return torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size, device=device)
        else:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_units=256, batch_size=64):
        """
        :param vocab_size: 翻译目标语言词典大小
        :param embedding_dim: 目标语言编码大小
        :param hidden_units: gru 隐藏层大小
        :param num_layers: gru 层数
        :param batch_size: 训练批次大小
        """
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_units,
                          num_layers=1,
                          bidirectional=False,
                          batch_first=True)
        self.fc = nn.Linear(in_features=hidden_units, out_features=vocab_size)
        # TODO softmax 的 dim 是不是正确的呢？
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x, h_0):
        """
        output.shape = (batch_size, seq_len, self.vocab_size)
        hidden.shape = (num_layers * bidirect, batch_size, self.hidden_units)
        :param x:
        :param h_0:
        :return: 最终的预测结果 output 和最后一个时间步的状态 hidden
        """
        embedded = self.embedding(x)
        embedded = F.relu(embedded)
        output, hidden = self.gru(embedded, h_0)
        # output shape (batch_size, seq_len, self.hidden_units)
        output = self.softmax(self.fc(output))
        return output, hidden


class AttDecoderRNN(nn.Module):
    """
    TODO Dropout, Masking
    """

    def __init__(self, vocab_size, embedding_dim=256, hidden_units=128, max_encoder_seq_length=50, batch_size=64, dropout_prob=0.1):
        super(AttDecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        self.hidden_units = hidden_units
        self.max_encoder_seq_length = max_encoder_seq_length
        self.batch_size = batch_size

        self.dropout = nn.Dropout(self.dropout_prob)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=hidden_units,
                          hidden_size=hidden_units,
                          batch_first=True,
                          bidirectional=False,
                          num_layers=1)
        self.attn = nn.Linear(in_features=(self.hidden_units + self.embedding_dim),
                              out_features=self.max_encoder_seq_length)
        self.attn_combine = nn.Linear(in_features=(self.embedding_dim + self.hidden_units),
                                      out_features=self.hidden_units)
        self.out = nn.Linear(in_features=self.hidden_units,
                             out_features=vocab_size)

    def forward(self, x, h_0, encoder_outputs):
        """
        :param x: shape = (batch_size, seq_len=1)
        :param h_0: shape = (num_layers * bidirect, batch_size, encoder_hidden_units)
        :param encoder_outputs: shape = (batch_size, seq_len, encoder_hidden_units)
        :return:
        """
        if h_0.shape[0] != 1:
            print('h_0.shape[0] is {}, exit()'.format(str(h_0.shape[1])))
            return None
        # (batch_size, seq_len=1, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        # attn_weights.shape = (batch_size, 1, encoder_max_len)
        attn_weights = F.softmax(self.attn(torch.cat([embedded, h_0.transpose(0, 1)], -1)), dim=-1)
        # attn_applied.shape = (batch_size, 1, encoder_hidden_units)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        output = torch.cat([embedded, attn_applied], -1)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output, h_0)
        # TODO log_softmax ?
        output = F.log_softmax(self.out(output), dim=-1)

        return output, hidden
