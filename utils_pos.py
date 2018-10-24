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
    n_pos = len(pos_tags)
    pos2id = {p: i for i, p in enumerate(pos_tags)}
    pos2id['その他'] = n_pos
    return pos2id


def get_onehotW(pos2id):
    n_pos = len(pos2id)
    onehotW = numpy.eye(n_pos - 1).astype(numpy.float32)
    empty_pos = numpy.zeros(n_pos - 1).astype(numpy.float32)
    onehotW = numpy.vstack((onehotW, empty_pos))
    return onehotW


def clean_pos(pos, pos_level):
    if pos_level == 1:
        return pos.split('-')[0]
    elif pos_level == 2:
        split = pos.split('-')
        return split[0] + '-' + split[1] if len(split) > 1 else pos
    elif pos_level == 3:
        split = pos.split('-')
        return split[0] + '-' + split[1] + '-' + split[2] if len(split) > 2 else pos

def make_pos_array(pos_tags, pos2id):
    return numpy.array([pos2id.get(p, pos2id['その他']) for p in pos_tags], numpy.int32)


def split_text_with_pos(lines, pos_level):
    left_words_data, right_words_data, targets_data = [], [], []
    left_pos_data, right_pos_data = [], []
    for line in tqdm(lines):
        line = line.replace('\n', '')
        line = clean_text(line)
        m = re.match(split_regex, line)
        left_text, target, right_text = m.groups()
        left_words = clean_text(left_text).split()
        right_words = clean_text(right_text).split()

        # lineのスペースを排除して形態素解析
        pos_tags = to_pos(line)  # [pos1, pos2, ...]
        pos_tags = [clean_pos(p, pos_level) for p in pos_tags]
        left_pos_tags, right_pos_tags = pos_tags[:len(left_words)], pos_tags[len(left_words) + 1:]

        if len(left_words) != len(left_pos_tags) or len(right_words) != len(right_pos_tags):
            # print('assert')
            continue

        left_words_data.append(left_words)
        right_words_data.append(right_words)
        targets_data.append(target)

        left_pos_data.append(left_pos_tags)
        right_pos_data.append(right_pos_tags)


    return left_words_data, right_words_data, targets_data, left_pos_data, right_pos_data


def make_dataset_with_pos(path_or_data, pos_level, w2id=None, class2id=None,
                          pos2id=None, pos2onehotW=None, vocab_size=40000, min_freq=1):
    if type(path_or_data) is list:
        lines = path_or_data
    else:
        lines = open(path_or_data, 'r', encoding='utf-8').readlines()
    lines = [line for line in lines if re.match(split_regex, line)]
    lines = [line for line in lines if len(line.split()) < 20]
    left_words, right_words, targets, left_pos, right_pos = split_text_with_pos(lines, pos_level)

    if not w2id and not class2id and not pos2id and not pos2onehotW:
        words = [w for words in left_words for w in words] + [w for words in right_words for w in words]
        w2id = get_vocab(words, vocab_size, min_freq)
        class2id = get_class(targets)
        pos_tags = [p for pos_tags in left_pos for p in pos_tags] + [p for pos_tags in right_pos for p in pos_tags]
        pos2id = get_pos(pos_tags)
        pos2onehotW = get_onehotW(pos2id)

    converters = {}
    converters['w2id'] = w2id
    converters['class2id'] = class2id
    converters['pos2id'] = pos2id
    converters['pos2onehotW'] = pos2onehotW

    left_arrays = [make_context_array(words, w2id) for words in left_words]
    right_arrays = [make_context_array(words, w2id) for words in right_words]
    target_arrays = [make_target_array(t, class2id) for t in targets]

    left_pos_arrays = [make_pos_array(pos_tags, pos2id) for pos_tags in left_pos]
    right_pos_arrays = [make_pos_array(pos_tags, pos2id) for pos_tags in right_pos]

    dataset = [(left_array, right_array, target_array, left_pos_array, right_pos_array, len(pos2onehotW) - 1)
              for left_array, right_array, target_array, left_pos_array, right_pos_array
              in zip(left_arrays, right_arrays, target_arrays, left_pos_arrays, right_pos_arrays)]

    return dataset, converters
