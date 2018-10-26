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
    pos2id['UNK'] = n_pos
    return pos2id


def clean_pos(pos, pos_level):
    if pos_level == 1:  # 大分類
        return pos.split('-')[0]
    elif pos_level == 2:  # 中分類
        return '-'.join(pos.split('-')[:2])
    elif pos_level == 3:  # 小分類
        return '-'.join(pos.split('-')[:3])
    elif pos_level == 4:  # マイ分類
        split = pos.split('-')
        if split[0] == '感動詞':
            return '感動詞'
        elif split[0] == '形状詞':
            return pos if split[1] == '助動詞語幹' else '形状詞-一般'
        elif split[0] in ['記号', '補助記号', '空白']:
            if len(split) > 1 and split[1] == '句点':
                return '句点'
            elif len(split) > 1 and split[1] == '読点':
                return '読点'
            else:
                return '記号'
        else:
            return '-'.join(split[:3])


def make_pos_array(pos_tags, pos2id):
    return numpy.array([pos2id.get(p, pos2id['UNK']) for p in pos_tags], numpy.int32)


def split_text_with_pos(lines, pos_level):
    left_words_data, right_words_data, targets_data = [], [], []
    left_pos_data, right_pos_data = [], []

    for line in tqdm(lines):
        line = line.replace('\n', '')
        line = clean_text(line)
        m = re.match(split_regex, line)
        if not m:
            continue
        left_text, target, right_text = m.groups()
        left_words = clean_text(left_text).split()
        right_words = clean_text(right_text).split()

        # lineのスペースを排除して形態素解析
        pos_tags = to_pos(line)  # [pos1, pos2, ...]
        pos_tags = [clean_pos(p, pos_level) for p in pos_tags]
        left_pos_tags, right_pos_tags = pos_tags[:len(left_words)], pos_tags[len(left_words) + 1:]

        if len(left_words) != len(left_pos_tags) or len(right_words) != len(right_pos_tags):
            continue

        left_words_data.append(left_words)
        right_words_data.append(right_words)
        targets_data.append(target)
        left_pos_data.append(left_pos_tags)
        right_pos_data.append(right_pos_tags)

    return left_words_data, right_words_data, targets_data, left_pos_data, right_pos_data


def make_dataset_with_pos(path_or_data, pos_level, w2id=None, class2id=None, pos2id=None, vocab_size=40000, min_freq=1):
    if type(path_or_data) is list:
        lines = path_or_data
    else:
        lines = open(path_or_data, 'r', encoding='utf-8').readlines()
    left_words, right_words, targets, left_pos, right_pos = split_text_with_pos(lines, pos_level)

    if not w2id and not class2id and not pos2id:
        words = [w for words in left_words for w in words] + [w for words in right_words for w in words]
        w2id = get_vocab(words, vocab_size, min_freq)
        class2id = get_class(targets)
        pos_tags = [p for pos_tags in left_pos for p in pos_tags] + [p for pos_tags in right_pos for p in pos_tags]
        pos2id = get_pos(pos_tags)

    dataset = [
        (make_context_array(lxs, w2id), make_context_array(rxs, w2id), make_target_array(t, class2id),
         make_pos_array(lps, pos2id), make_pos_array(rps, pos2id))
        for lxs, rxs, t, lps, rps
        in zip(left_words, right_words, targets, left_pos, right_pos)
    ]
    converters = {'w2id': w2id, 'class2id': class2id, 'pos2id': pos2id}

    return dataset, converters
