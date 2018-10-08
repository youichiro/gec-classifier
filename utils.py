import re
import mojimoji
import numpy
from tqdm import tqdm
from collections import Counter


IGNORE_ID = -1
UNK_ID = 0
split_regex = r'^(.*) <(.)> (.*)$'
digit_regex = re.compile(r'(\d( \d)*)+')


def clean_text(text):
    text = mojimoji.zen_to_han(text, kana=False)
    text = digit_regex.sub('#', text)
    return text


def get_vocab(words, vocab_size, min_freq):
    counter = Counter()
    for w in words:
        counter[w] += 1
    w2id = {w: i for i, (w, f) in enumerate(counter.most_common(vocab_size), 1) if f >= min_freq}
    w2id['UNK'] = UNK_ID
    return w2id


def get_class(targets):
    targets = set(targets)
    classes = {t: i for i, t in enumerate(targets)}
    return classes


def make_context_array(words, w2id):
    ids = [w2id.get(w, UNK_ID) for w in words]
    return numpy.array(ids, numpy.int32)


def make_target_array(target, classes):
    return numpy.array([classes[target]], numpy.int32)


def split_text(lines):
    left_words, right_words, targets = [], [], []
    for line in tqdm(lines):
        m = re.match(split_regex, line.rstrip())
        if not m:
            continue
            #TODO: create_dataset.pyで文末の格助詞タグを付けないようにして再実行
        left_text, target, right_text = m.groups()
        left_words.append(clean_text(left_text).split())
        right_words.append(clean_text(right_text).split())
        targets.append(target)
    return left_words, right_words, targets


def make_dataset(path, w2id=None, classes=None, vocab_size=40000, min_freq=1):
    """
    example return:
        dataset = [
            (numpy.array([1, 2, 3, 1]), numpy.array([1, 2, 3]), numpy.array([1])),
            (numpy.array([1, 5, 0, 2, 5]), numpy.array([1, 2, 3, 4]), numpy.array([1])),
            (numpy.array([1, 2, 3]), numpy.array([1, 5]), numpy.array([1])),
            ...
        ]
        w2id = {'UNK': 0, 'token1': 1, 'token2': 2, ...}
        classes = {'class1': 0, 'class2': 1, 'class3': 2, ...}
    """
    lines = open(path, 'r', encoding = 'utf-8').readlines()
    left_words, right_words, targets = split_text(lines)

    if not w2id or not classes:
        words = [w for words in left_words for w in words] + [w for words in right_words for w in words]
        w2id = get_vocab(words, vocab_size, min_freq)
        classes = get_class(targets)

    left_arrays = [make_context_array(words, w2id) for words in left_words]
    right_arrays = [make_context_array(words, w2id) for words in right_words]
    target_arrays = [make_target_array(t, classes) for t in targets]

    dataset = [(left_array, right_array, target_array)
               for left_array, right_array, target_array in zip(left_arrays, right_arrays, target_arrays)]

    return dataset, w2id, classes
