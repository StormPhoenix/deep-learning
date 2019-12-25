import nlp.couplet_generator.config as config
from nlp.couplet_generator.generator import SimpleSeq2SeqGenerator
from nlp.couplet_generator.utils.data_utils import build_vocab

if config.use_model == 'attention-seq2seq':
    from nlp.couplet_generator.seq2seq.models import AttentionalSeq2seqModel as Seq2SeqModel
elif config.use_model == 'simple-seq2seq':
    from nlp.couplet_generator.seq2seq.models import SimpleSeq2SeqModel as Seq2SeqModel


def main():
    word_id_map, id_word_map = build_vocab()
    generator = SimpleSeq2SeqGenerator(config.train_set_size, config.batch_size, word_id_map)
    model = Seq2SeqModel(len(word_id_map), word_id_map.get_id('<pad>', 0))
    model.train(generator)
    while True:
        sententce = input('input:')
        predicted_result = model.predict(sententce, word_id_map, id_word_map)
        print(predicted_result)


if __name__ == '__main__':
    main()
