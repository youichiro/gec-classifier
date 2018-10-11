import re
import mojimoji
import numpy
import random
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
    class2id = {t: i for i, t in enumerate(targets)}
    return class2id


def make_context_array(words, w2id):
    ids = [w2id.get(w, UNK_ID) for w in words]
    return numpy.array(ids, numpy.int32)


def make_target_array(target, class2id):
    return numpy.array([class2id[target]], numpy.int32)


def split_text(lines):
    left_words, right_words, targets = [], [], []
    for line in tqdm(lines):
        m = re.match(split_regex, line.replace('\n', ''))
        if not m:
            continue
            #TODO: create_dataset.pyで文末の格助詞タグを付けないようにして再実行
        left_text, target, right_text = m.groups()
        left_words.append(clean_text(left_text).split())
        right_words.append(clean_text(right_text).split())
        targets.append(target)
    return left_words, right_words, targets


def make_dataset(path_or_data, w2id=None, class2id=None, vocab_size=40000, min_freq=1):
    """
    example return:
        dataset = [
            (numpy.array([1, 2, 3, 1]), numpy.array([1, 2, 3]), numpy.array([1])),
            (numpy.array([1, 5, 0, 2, 5]), numpy.array([1, 2, 3, 4]), numpy.array([1])),
            (numpy.array([1, 2, 3]), numpy.array([1, 5]), numpy.array([1])),
            ...
        ]
        w2id = {'UNK': 0, 'token1': 1, 'token2': 2, ...}
        class2id = {'class1': 0, 'class2': 1, 'class3': 2, ...}
    """
    if type(path_or_data) is list:
        lines = path_or_data
    else:
        lines = open(path_or_data, 'r', encoding = 'utf-8').readlines()
    left_words, right_words, targets = split_text(lines)

    if not w2id or not class2id:
        words = [w for words in left_words for w in words] + [w for words in right_words for w in words]
        w2id = get_vocab(words, vocab_size, min_freq)
        class2id = get_class(targets)

    left_arrays = [make_context_array(words, w2id) for words in left_words]
    right_arrays = [make_context_array(words, w2id) for words in right_words]
    target_arrays = [make_target_array(t, class2id) for t in targets]

    dataset = [(left_array, right_array, target_array)
               for left_array, right_array, target_array in zip(left_arrays, right_arrays, target_arrays)]

    return dataset, w2id, class2id


def tagging(err, ans):
    """
    err: 分かち書きされた誤り文
    ans: 分かち書きされた正解文
    条件: len(err) == len(ans) and err != ans
    return: errとansの不一致箇所の1つをタグ(<>)付けした文とerror
    """
    diff_ids = [i for i in range(len(err)) if err[i] != ans[i]]
    idx = diff_ids[0] if len(diff_ids) == 1 else random.choice(diff_ids)
    test = ans[:idx] + '<' + ans[idx] + '>' + ans[idx+1:]
    return test

