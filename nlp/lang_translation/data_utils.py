import nlp.lang_translation.config as config
from nlp.lang_translation.lang import Lang


def empty(string: str):
    return string is None or len(string) is 0 or string.isspace()


def filter_chinese_texts(text_line: str):
    filter_words = []
    words = text_line.split(' ')

    for index, word in enumerate(words):
        if index == 0:
            if word.isdigit() and len(words) > 2 and words[1] == '.':
                continue

        if index == 1:
            if word == '.' and words[0].isdigit():
                continue

        if word == '':
            continue

        if index == 1 and word == '.':
            continue

        if index == len(words) - 1 and (word == '\n' or word == '.\n' or word == '。\n'):
            continue

        if index == len(words) - 1 and word.endswith('\n'):
            word = word[:-1]

        if index == len(words) - 1 and word.endswith('。\n'):
            word = word[:word.index('。\n')]

        filter_words.append(word)
    return filter_words


def filter_english_texts(text_line: str):
    filter_words = []
    words = text_line.split(' ')
    for index, word in enumerate(words):
        if word == '':
            continue

        if index == len(words) - 1 and word.endswith('.\n'):
            word = word[:word.index('.\n')]

        if index == len(words) - 1 and word.endswith('\n'):
            word = word[:word.index('\n')]

        if word.endswith('\'s'):
            word = word[:word.index('\'s')]
            filter_words.append(word.lower())
            filter_words.append('is')
        else:
            filter_words.append(word.lower())
    return filter_words


def read_texts(path):
    with open(path, 'r', encoding='utf8') as fr:
        texts = fr.readlines()
    return texts


def read_chinese_texts():
    texts = read_texts(config.CHINESE_LANG_PATH)

    filter_texts = []
    for line in texts:
        filter_words = filter_chinese_texts(line)
        filter_texts.append(filter_words)
    return filter_texts


def read_english_texts():
    texts = read_texts(config.ENGLISH_LANG_PATH)

    filter_texts = []
    for line in texts:
        filter_words = filter_english_texts(line)
        filter_texts.append(filter_words)
    return filter_texts


def get_lang_dict():
    """
    从文本中获取双语字典
    :return:
    """
    ch_lang = Lang('chinese')
    en_lang = Lang('english')

    if ch_lang.is_dict_exists():
        ch_lang.load()
    else:
        print('reading chinese text')
        # 经过过滤
        chinese_texts = read_chinese_texts()
        for words in chinese_texts:
            ch_lang.add_sentence(words)
        ch_lang.save()

    if en_lang.is_dict_exists():
        en_lang.load()
    else:
        print('reading english text')
        english_texts = read_english_texts()
        for words in english_texts:
            en_lang.add_sentence(words)
        en_lang.save()

    print('get_lang_dict()')
    return ch_lang, en_lang


def couplets_generator(train_set_size=20000,
                       couplet_in_path='./resources/data/couplet/dev/clean_in.txt',
                       couplet_out_path='./resources/data/couplet/dev/clean_out.txt'):
    while True:
        fr_in = open(couplet_in_path, "r", encoding='utf8')
        fr_out = open(couplet_out_path, "r", encoding='utf8')

        line_in = fr_in.readline()
        line_out = fr_out.readline()
        count = 0
        while not empty(line_in) and not empty(line_out):
            if line_in.find('couplets') is not -1 or line_out.find('couplets') is not -1:
                line_in = fr_in.readline()
                line_out = fr_out.readline()
                continue
            count += 1
            line_in = (line_in[:-1])
            line_out = (line_out[:-1])
            yield line_in, line_out
            if count > train_set_size:
                break
            line_in = fr_in.readline()
            line_out = fr_out.readline()
        fr_in.close()
        fr_out.close()


def get_text_pairs():
    ch_lang_dict, en_lang_dict = get_lang_dict()

    # 经过过滤的文本
    chinese_texts = read_chinese_texts()
    english_texts = read_english_texts()

    parsed_chinese_texts = []
    parsed_english_texts = []

    for row in range(len(chinese_texts)):
        chinese_words = chinese_texts[row]
        english_words = english_texts[row]

        if len(chinese_words) > config.max_length or len(english_words) > config.max_length:
            continue

        parsed_chinese_words = ch_lang_dict.parse_words_to_ids(chinese_words)
        parsed_english_words = en_lang_dict.parse_words_to_ids(english_words)

        parsed_chinese_texts.append(parsed_chinese_words)
        parsed_english_texts.append(parsed_english_words)

    return (parsed_chinese_texts, parsed_english_texts), ch_lang_dict, en_lang_dict


if __name__ == '__main__':
    get_text_pairs()

    print('bingo')
