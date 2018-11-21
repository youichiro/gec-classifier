import re
import mojimoji
import numpy
import random
from tqdm import tqdm
from collections import Counter
import chainer.computational_graph as c


IGNORE_ID = -1
UNK_ID = 0
split_regex = r'^(.*) <(.)> (.*)$'
digit_regex = re.compile(r'(\d( \d)*)+')


def clean_text(text):
    text = mojimoji.zen_to_han(text, kana=False)
    text = digit_regex.sub('#', text)
    return text


def get_vocab(words, vocab_size, min_freq):
    words += ['<TARGET>']
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
        left_text, target, right_text = m.groups()
        left_words.append(clean_text(left_text).split())
        right_words.append(clean_text(right_text).split())
        targets.append(target)
    return left_words, right_words, targets


def make_dataset(path_or_data, w2id=None, class2id=None, vocab_size=40000, min_freq=1, n_encoder=2):
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
        lines = open(path_or_data, 'r', encoding='utf-8').readlines()
    lines = [line for line in lines if re.match(split_regex, line)]
    left_words, right_words, targets = split_text(lines)

    if not w2id or not class2id:
        words = [w for words in left_words for w in words] + [w for words in right_words for w in words]
        w2id = get_vocab(words, vocab_size, min_freq)
        class2id = get_class(targets)

    if n_encoder == 2:
        dataset = [
            (make_context_array(lxs, w2id), make_context_array(rxs, w2id), make_target_array(t, class2id))
            for lxs, rxs, t
            in zip(left_words, right_words, targets)
        ]
    else:
        dataset = [
            (numpy.concatenate( (make_context_array(lxs, w2id),
                                 numpy.array([w2id['<TARGET>']], numpy.int32),
                                 make_context_array(rxs, w2id)) ),
             make_target_array(t, class2id) )
            for lxs, rxs, t
            in zip(left_words, right_words, targets)
        ]
    converters = {'w2id': w2id, 'class2id': class2id}

    return dataset, converters


def tagging(err, ans):
    """
    err: 分かち書きされた誤り文
    ans: 分かち書きされた正解文
    条件: len(err) == len(ans) and err != ans
    return: errとansの不一致箇所の1つをタグ(<>)付けした文とerror
    """
    diff_ids = [i for i in range(len(err)) if err[i] != ans[i]]
    idx = diff_ids[0] if len(diff_ids) == 1 else diff_ids[1]
    test = ans[:idx] + '<' + ans[idx] + '>' + ans[idx+1:]
    return test


def graph(model):
    """モデルのネットワークグラフを描写する"""
    lxs = numpy.array([[1, 2, 3], [7, 8, 9]])
    rxs = numpy.array([[4, 5, 6], [10, 11, 12]])
    ts = numpy.array([[1], [2]])
    lps = numpy.array([[1, 2, 3], [7, 8, 9]])
    rps = numpy.array([[4, 5, 6], [10, 11, 12]])
    loss = model(lxs, rxs, ts, lps, rps)
    g = c.build_computational_graph([loss])
    with open('/graph.dot', 'w') as o:
        o.write(g.dump())
    print('Has witten graph.dot')
