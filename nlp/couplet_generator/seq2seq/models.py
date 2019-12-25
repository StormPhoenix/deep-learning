import os

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, Activation, Dot, RepeatVector, \
    Concatenate, Reshape, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model

import nlp.couplet_generator.config as config
from nlp.couplet_generator.callback import LossHistory


class SimpleSeq2SeqModel:
    def __init__(self, vocab_size, mask_id):
        self.mask_id = mask_id
        # TODO clean注释
        # 对于非定长编码，这个地方是需要用 T_X 还是 None
        self.encoder_inputs = Input(shape=(config.couplet_max_len,), name='encoder_inputs')
        # self.encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        self.decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        # self.decoder_inputs = Input(shape=(config.couplet_max_len + 1,), name='decoder_inputs')

        self.encoder_masking = Masking(mask_value=self.mask_id, input_shape=(config.couplet_max_len,))
        self.encoder_embedding = Embedding(input_dim=vocab_size, output_dim=config.enc_emb_dim)
        self.encode_lstm = LSTM(units=config.enc_lstm_hidden_dim, return_state=True)

        self.decoder_masking = Masking(mask_value=self.mask_id)
        # self.decoder_masking = Masking(mask_value=self.mask_id, input_shape=(config.couplet_max_len + 1,))
        self.decoder_embedding = Embedding(input_dim=vocab_size, output_dim=config.dec_emb_dim)
        self.decoder_lstm = LSTM(units=config.dec_lstm_hidden_dim, return_sequences=True, return_state=True,
                                 name='decoder_lstm')
        self.decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))

        self.model: Model = None
        self.encoder: Model = None
        self.decoder: Model = None

        self.build_model()

    def encode(self):
        enc_masking = self.encoder_masking(self.encoder_inputs)
        # enc_emb_x.shape = (batch, seq_len, encoder_embedding_output_dim)
        enc_emb_x = self.encoder_embedding(enc_masking)
        # encoder_outputs.shape = (batch, enc_lstm_hidden_dim)
        # state_c.shape = (batch, encoder_embedding_output_dim)
        # state_h.shape = (batch, encoder_embedding_output_dim)
        encoder_outputs, state_h, state_c = self.encode_lstm(enc_emb_x)
        encode_states = [state_h, state_c]
        return encoder_outputs, encode_states

    def decode(self, encode_outputs, encode_states):
        dec_masking = self.decoder_masking(self.decoder_inputs)
        # dec_emb_x.shape = (batch, seq_len, dec_emb_dim)
        dec_emb_x = self.decoder_embedding(dec_masking)
        # decode_lstm_outputs.shape = (batch, seq_len, dec_lstm_hidden_dim)
        decode_lstm_outputs, _, _ = self.decoder_lstm(dec_emb_x, initial_state=encode_states)
        # decode_outputs.shape = (batch, seq_len, vocab_size)
        decode_outputs = self.decoder_dense(decode_lstm_outputs)
        return decode_outputs

    def build_model(self):
        encoder_outputs, encoder_states = self.encode()
        decoder_outputs = self.decode(encoder_outputs, encoder_states)
        # build whole models
        self.model = Model(inputs=[self.encoder_inputs, self.decoder_inputs], outputs=decoder_outputs)

        if config.log_model_info:
            plot_model(self.model, to_file=config.model_image_path, show_shapes=True)
        # build encoder models
        self.encoder = Model(inputs=self.encoder_inputs, outputs=encoder_states)

        # build decoder models
        decoder_state_h_input = Input(shape=(config.dec_lstm_hidden_dim,), name='decoder_state_h_input')
        decoder_state_c_input = Input(shape=(config.dec_lstm_hidden_dim,), name='decoder_state_c_input')
        decoder_states_input = [decoder_state_h_input, decoder_state_c_input]

        decoder_lstm_outputs, decoder_state_h, decoder_state_c = self.decoder_lstm(
            self.decoder_embedding(self.decoder_inputs),
            initial_state=decoder_states_input)
        decoder_states = [decoder_state_h, decoder_state_c]
        decoder_outputs = self.decoder_dense(decoder_lstm_outputs)
        self.decoder = Model(inputs=[self.decoder_inputs] + decoder_states_input,
                             outputs=[decoder_outputs] + decoder_states)

    def train(self, generator, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch,
              checkpoints_path=config.check_points_path):
        self.model.compile(optimizer=Adam(lr=0.009, beta_1=0.9, beta_2=0.999),
                           metrics=['accuracy'],
                           loss='categorical_crossentropy')
        if os.path.exists(config.weight_model_path):
            self.model.load_weights(config.weight_model_path)
            self.encoder.load_weights(config.weight_encoder_path)
            self.decoder.load_weights(config.weight_decoder_path)

        if not os.path.exists(config.weight_model_path) or config.insist_training:
            callback = ModelCheckpoint(filepath=checkpoints_path, monitor='loss', save_best_only=True,
                                       save_weights_only=True, period=100)
            history_loss_callback = LossHistory()
            self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, workers=1, epochs=epochs,
                                     callbacks=[callback, history_loss_callback])
            self.model.save_weights(config.weight_model_path)
            self.encoder.save_weights(config.weight_encoder_path)
            self.decoder.save_weights(config.weight_decoder_path)
            history_loss_callback.loss_plot('epoch')

    def predict(self, input_sentence, word2id_map, id2word_map):
        if not os.path.exists(config.weight_model_path):
            raise ValueError('Model must be trained before predicting. ')

        input_sentence = [word2id_map.get_id(word, 1) for word in input_sentence]
        input_sentence = pad_sequences([input_sentence], maxlen=config.couplet_max_len, padding='post',
                                       value=word2id_map.get_id('<pad>', 0))

        states_value = self.encoder.predict(np.array(input_sentence))

        target_sentence = np.zeros(shape=(1, 1))
        target_sentence[0, 0] = word2id_map.get_id('<go>', 0)

        decoded_sentence = ''
        while True:
            # output.shape = (batch, seq_len, vocab_size) = (1, 1, vocab_size)
            output, states_h, states_c = self.decoder.predict([target_sentence] + states_value)
            token_index = np.argmax(output[0, 0, :])

            if token_index == word2id_map.get_id('<eos>', 0):
                break

            token = id2word_map.get_id(token_index, '<unk>')
            decoded_sentence += token
            target_sentence[0, 0] = token_index
            states_value = [states_h, states_c]
        return decoded_sentence


class AttentionalSeq2seqModel:
    def __init__(self, vocab_size, mask_id):
        self.mask_id = mask_id
        # 对于非定长编码，这个地方是需要用 T_X 还是 None
        # Encoder
        self.encoder_inputs = Input(shape=(config.couplet_max_len,), name='encoder_inputs')
        self.decoder_inputs = Input(shape=(config.couplet_max_len + 1,), name='decoder_inputs')

        self.encoder_masking = Masking(mask_value=self.mask_id, input_shape=(config.couplet_max_len,))
        self.encoder_embedding = Embedding(input_dim=vocab_size, output_dim=config.enc_emb_dim)
        self.encoder_lstm = LSTM(units=config.enc_lstm_hidden_dim, return_sequences=True, return_state=True)

        # Decoder
        self.decoder_masking = Masking(mask_value=self.mask_id)
        # self.decoder_masking = Masking(mask_value=self.mask_id, input_shape=(config.couplet_max_len + 1,))
        self.decoder_embedding = Embedding(input_dim=vocab_size, output_dim=config.dec_emb_dim)
        self.decoder_reshape = Reshape(target_shape=(1, config.dec_emb_dim))
        self.decoder_concat = Concatenate(axis=-1, name='decoder_concat')
        self.decoder_lstm = LSTM(units=config.dec_lstm_hidden_dim, return_state=True,
                                 name='decoder_lstm')
        self.decoder_dense = Dense(vocab_size, activation='softmax')

        # Attention
        self.attention_repeat = RepeatVector(config.couplet_max_len, name='attention_repeat')
        self.attention_concat = Concatenate(axis=-1, name='attention_concat')
        self.attention_dense_1 = Dense(units=config.attention_dense_1_units, activation='tanh',
                                       name='attention_dense_1')
        self.attention_dense_2 = Dense(units=1, activation='relu', name='attention_dense_2')
        self.attention_activation = Activation(activation='softmax', name='attention_activation')
        self.attention_dot = Dot(axes=1)

        self.model: Model = None
        self.encoder: Model = None
        self.decoder: Model = None

        self.build()

    def encode(self):
        enc_masking = self.encoder_masking(self.encoder_inputs)
        enc_emb_x = self.encoder_embedding(enc_masking)
        encoder_outputs, state_h, state_c = self.encoder_lstm(enc_emb_x)
        return encoder_outputs, state_h, state_c

    def attention(self, enc_outputs, dec_pre_hidden):
        # enc_outputs.shape = (batch, seq_len, enc_emb_dim)
        dec_pre_hidden = self.attention_repeat(dec_pre_hidden)
        # dec_pre_hidden.shape = (batch, seq_len, dec_hidden_dim)
        concat = self.attention_concat([dec_pre_hidden, enc_outputs])
        dense1 = self.attention_dense_1(concat)
        dense2 = self.attention_dense_2(dense1)
        weights = self.attention_activation(dense2)
        context = self.attention_dot([weights, enc_outputs])
        return context

    def decode(self, concat_input, initial_states):
        decode_lstm_output, state_h, state_c = self.decoder_lstm(concat_input, initial_state=initial_states)
        # decode_outputs.shape = (batch, seq_len, vocab_size)
        decode_output = self.decoder_dense(decode_lstm_output)
        return decode_output, state_h, state_c

    def build_model(self):
        slice = lambda x, index: x[:, index, :]
        outputs = []
        enc_outputs, enc_output_state_h, enc_output_state_c = self.encode()
        dec_state_h = enc_output_state_h
        dec_state_c = enc_output_state_c
        for i in range(config.couplet_max_len + 1):
            # attention context
            context = self.attention(enc_outputs, dec_state_h)
            # concat context and input
            reshaped_input = self.decoder_reshape(Lambda(slice, arguments={'index': i})(
                self.decoder_embedding(self.decoder_masking(self.decoder_inputs))))
            concat = self.decoder_concat([context, reshaped_input])
            initial_states = [dec_state_h, dec_state_c]
            output, dec_state_h, dec_state_c = self.decode(concat, initial_states)
            outputs.append(output)

        # build whole models
        self.model = Model(inputs=[self.encoder_inputs, self.decoder_inputs], outputs=outputs)
        if config.log_model_info:
            plot_model(self.model, to_file=config.model_image_path, show_shapes=True)

    def build_encoder(self):
        # 如果 encoder 的 input 序列长度设置为 None，那么就无法进行批量预测
        enc_output, enc_state_h, enc_state_c = self.encode()
        self.encoder = Model(inputs=self.encoder_inputs, outputs=[enc_output, enc_state_h, enc_state_c])

    def build_decoder(self):
        dec_input = Input(shape=(1,))
        dec_state_h_input = Input(shape=(config.dec_lstm_hidden_dim,), name='dec_state_h_input')
        dec_state_c_input = Input(shape=(config.dec_lstm_hidden_dim,), name='dec_state_c_input')
        dec_states_input = [dec_state_h_input, dec_state_c_input]
        # define encoder_outputs input
        enc_outputs = Input(shape=(config.couplet_max_len, config.enc_lstm_hidden_dim))

        context = self.attention(enc_outputs, dec_state_h_input)
        embedded_input = self.decoder_embedding(self.decoder_masking(dec_input))
        concat = self.decoder_concat([context, embedded_input])
        output, state_h, state_c = self.decode(concat, dec_states_input)
        self.decoder = Model(inputs=[dec_input, enc_outputs] + dec_states_input, outputs=[output, state_h, state_c])

    def build(self):
        self.build_model()
        self.build_encoder()
        self.build_decoder()

    def train(self, generator, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch,
              checkpoints_path=config.check_points_path):
        self.model.compile(optimizer=Adam(lr=0.009, beta_1=0.9, beta_2=0.999),
                           metrics=['accuracy'],
                           loss='categorical_crossentropy')
        if os.path.exists(config.weight_model_path):
            self.model.load_weights(config.weight_model_path)
            self.encoder.load_weights(config.weight_encoder_path)
            self.decoder.load_weights(config.weight_decoder_path)

        if not os.path.exists(config.weight_model_path) or config.insist_training:
            checkpoints_callback = ModelCheckpoint(filepath=checkpoints_path, monitor='loss', save_best_only=True,
                                                   save_weights_only=True, period=5)
            history_loss_callback = LossHistory()
            self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, workers=1, epochs=epochs,
                                     callbacks=[checkpoints_callback, history_loss_callback])
            self.model.save_weights(config.weight_model_path)
            self.encoder.save_weights(config.weight_encoder_path)
            self.decoder.save_weights(config.weight_decoder_path)

            history_loss_callback.loss_plot('epoch')

    def predict(self, input_sentence, word2id_map, id2word_map):
        if not os.path.exists(config.weight_model_path):
            raise ValueError('Model must be trained before predicting. ')

        input_sentence = [word2id_map.get_id(word, 1) for word in input_sentence]
        input_sentence = pad_sequences([input_sentence], maxlen=config.couplet_max_len, padding='post',
                                       value=word2id_map.get_id('<pad>', 0))

        enc_output, enc_state_h, enc_state_c = self.encoder.predict(np.array(input_sentence))

        target_sentence = np.zeros(shape=(1, 1))
        target_sentence[0, 0] = word2id_map.get_id('<go>', 0)

        decoded_sentence = ''
        state_h = enc_state_h
        state_c = enc_state_c
        while True:
            output, state_h, state_c = self.decoder.predict([target_sentence, enc_output, state_h, state_c])

            token_index = np.argmax(output[0, :])

            if token_index == word2id_map.get_id('<eos>', 0):
                break

            token = id2word_map.get_id(token_index, '<unk>')
            decoded_sentence += token
            target_sentence[0, 0] = token_index
        return decoded_sentence
