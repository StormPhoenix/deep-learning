import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import nlp.couplet_generator.config as config
from nlp.couplet_generator.utils.data_utils import couplets_generator


class SimpleSeq2SeqGenerator(keras.utils.Sequence):
    def __init__(self, train_set_size: int, batch_size: int, vocabulary: dict):
        self.train_set_size = train_set_size
        self.batch_size = batch_size
        self.word_id_map = vocabulary
        self.generator = couplets_generator(self.train_set_size)
        self.pad_id = self.word_id_map.get('<pad>', 0)
        self.go_id = self.word_id_map.get('<go>', 2)
        self.eos_id = self.word_id_map.get('<eos>', 3)

    def __len__(self):
        return int(self.train_set_size / self.batch_size)

    def __getitem__(self, item):
        encoder_inputs = []
        decoder_inputs = []
        decoder_outputs = []
        count = 0
        while count < self.batch_size:
            couplet_in, couplet_out = next(self.generator)
            if len(couplet_in) is not len(couplet_out) or len(couplet_in) > config.couplet_max_len:
                continue
            count += 1
            enc_input = [self.word_id_map.get(word, 1) for word in couplet_in]

            couplet_out_id = [self.word_id_map.get(word, 1) for word in couplet_out]
            dec_input = [self.go_id] + couplet_out_id

            dec_output = couplet_out_id + [self.eos_id]

            encoder_inputs.append(enc_input)
            decoder_inputs.append(dec_input)
            decoder_outputs.append(dec_output)
        encoder_inputs = pad_sequences(encoder_inputs, maxlen=config.couplet_max_len, padding='post', value=self.pad_id)
        decoder_inputs = pad_sequences(decoder_inputs, maxlen=config.couplet_max_len + 1, padding='post',
                                       value=self.pad_id)
        decoder_outputs = pad_sequences(decoder_outputs, maxlen=config.couplet_max_len + 1, padding='post',
                                        value=self.eos_id)
        decoder_outputs = np.array(
            list(map(lambda x: to_categorical(x, num_classes=len(self.word_id_map)), decoder_outputs)))

        encoder_inputs = np.array(encoder_inputs)
        decoder_inputs = np.array(decoder_inputs)
        # decoder_outputs = np.array(decoder_outputs)
        if config.use_model == 'attention-seq2seq':
            decoder_outputs = list(decoder_outputs.swapaxes(0, 1))
        else:
            decoder_outputs = np.array(decoder_outputs)
        return [encoder_inputs, decoder_inputs], decoder_outputs
