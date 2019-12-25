import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

import nlp.couplet_generator.config as config
from nlp.couplet_generator.generator import SimpleSeq2SeqGenerator
from nlp.couplet_generator.utils.data_utils import build_vocab, load_clean_couplet_data

if config.use_model == 'attention-seq2seq':
    from nlp.couplet_generator.seq2seq.models import AttentionalSeq2seqModel as Seq2SeqModel
elif config.use_model == 'simple-seq2seq':
    from nlp.couplet_generator.seq2seq.models import SimpleSeq2SeqModel as Seq2SeqModel


def main():
    word_id_map, id_word_map = build_vocab()
    generator = SimpleSeq2SeqGenerator(config.train_set_size, config.batch_size, word_id_map)
    model = Seq2SeqModel(len(word_id_map), word_id_map.get_id('<pad>', 0))
    model.train(generator)

    couplet_in, couplet_out = load_clean_couplet_data(config.couplet_path['test']['in'],
                                                      config.couplet_path['test']['out'])

    result = pd.DataFrame(columns=['couplets'])

    smooth_func = SmoothingFunction()
    test_size = couplet_in.shape[0]
    total_score = 0
    for index, row in couplet_in.iterrows():
        sentence = row['couplets']
        prediction = model.predict(sentence, word_id_map, id_word_map)

        reference = couplet_out.iloc[1]['couplets']
        total_score += sentence_bleu([reference], prediction, smoothing_function=smooth_func.method7)

        result = result.append(pd.DataFrame({'couplets': [prediction]}), ignore_index=True)

    print('total score: ', total_score)
    print('average score: ', total_score / float(test_size))

    generated_path = './resources/data/generated/seq2seq_attention.txt'
    # generated_path = './resources/data/generated/simple_seq2seq.txt'
    result.to_csv(generated_path, index=None, header=None, encoding='utf8')


if __name__ == '__main__':
    main()
