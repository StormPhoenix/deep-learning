use_model = 'attention-seq2seq'
# use_model = 'simple-seq2seq'

if use_model == 'attention-seq2seq':
    from nlp.couplet_generator.config.config_attention_seq2seq import *
elif use_model == 'simple-seq2seq':
    from nlp.couplet_generator.config.config_simple_seq2seq import *

print(information)

log_model_info = False
