import re
import numpy
from tqdm import tqdm
from utils import (get_vocab, get_class, split_regex, clean_text,
                   make_context_array, make_target_array)
from mecab import Mecab

mecab_dict_file = '/tools/env/lib/mecab/dic/unidic'
mecab = Mecab(mecab_dict_file)


def to_pos(text):
    text = text.replace('<', '').replace('>', '').replace(' ', '')
    _, parts = mecab.tagger(text)
    return parts


def get_pos(pos_tags):
    pos_tags = list(set(pos_tags))
    pos2id = {p: i for i, p in enumerate(pos_tags)}
    return pos2id


def get_onehotW(n_pos):
    return numpy.eye(n_pos)


def clean_pos(pos):
    return pos.split('-')[0]


def split_text_with_pos(lines):
    left_words_data, right_words_data, targets_data = [], [], []
    left_pos_data, right_pos_data = [], []
    for line in tqdm(lines):
        m = re.match(split_regex, line.replace('\n', ''))
        left_text, target, right_text = m.groups()
        left_words = clean_text(left_text).split()
        right_words = clean_text(right_text).split()

        left_words_data.append(left_words)
        right_words_data.append(right_words)
        targets_data.append(target)

        # lineのスペースを排除して形態素解析
        pos_tags = to_pos(line)  # [pos1, pos2, ...]
        pos_tags = [clean_pos(p) for p in pos_tags]
        left_pos_tags, right_pos_tags = pos_tags[:len(left_words)], pos_tags[len(left_words) + 1:]
        left_pos_data.append(left_pos_tags)
        right_pos_data.append(right_pos_tags)
    return left_words_data, right_words_data, targets_data, left_pos_data, right_pos_data


def make_dataset_with_pos(path, w2id=None, class2id=None, pos2id=None, pos2onehotW=None, vocab_size=40000, min_freq=1):
    lines = open(path, 'r', encoding='utf-8').readlines()
    left_words, right_words, targets, left_pos, right_pos = split_text_with_pos(lines)

    if not w2id and not class2id and not pos2id and not pos2onehotW:
        words = [w for words in left_words for w in words] + [w for words in right_words for w in words]
        w2id = get_vocab(words, vocab_size, min_freq)
        class2id = get_class(targets)
        pos_tags = [p for pos_tags in left_pos for p in pos_tags] + [p for pos_tags in right_pos for p in pos_tags]
        pos2id = get_pos(pos_tags)
        pos2onehotW = get_onehotW(len(pos2id))

    converters = {}
    converters['w2id'] = w2id
    converters['class2id'] = class2id
    converters['pos2id'] = pos2id
    converters['pos2onehotW'] = pos2onehotW

    left_arrays = [make_context_array(words, w2id) for words in left_words]
    right_arrays = [make_context_array(words, w2id) for words in right_words]
    target_arrays = [make_target_array(t, class2id) for t in targets]

    left_pos_arrays = [[pos2onehotW[pos2id[p]] for p in pos_tags] for pos_tags in left_pos]
    right_pos_arrays = [[pos2onehotW[pos2id[p]] for p in pos_tags] for pos_tags in right_pos]

    dataset = [(left_array, right_array, target_array, left_pos_array, right_pos_array)
              for left_array, right_array, target_array, left_pos_array, right_pos_array
              in zip(left_arrays, right_arrays, target_arrays, left_pos_arrays, right_pos_arrays)]

    return dataset, converters
