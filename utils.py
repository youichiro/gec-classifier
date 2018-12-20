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
mecab_dict_file = '/tools/env/lib/mecab/dic/unidic'


def clean_text(text, to_kana=False):
    """全角→半角, 数字の正規化を行う"""
    text = mojimoji.zen_to_han(text, kana=False)
    text = digit_regex.sub('#', text)
    if to_kana:
        from mecab import Mecab
        mecab = Mecab(mecab_dict_file)
        text = ' '.join(mecab.to_kana(text))  # 平仮名に変換
    return text


def get_vocab(words, vocab_size, min_freq):
    """単語→IDの辞書を作成する"""
    counter = Counter()
    for w in words:
        counter[w] += 1
    w2id = {w: i for i, (w, f) in enumerate(counter.most_common(vocab_size), 1) if f >= min_freq}
    w2id['<UNK>'] = UNK_ID
    w2id['<TARGET>'] = len(w2id)
    return w2id


def get_class(targets):
    """分類する単語→IDの辞書を作成する"""
    targets = set(targets)
    class2id = {t: i for i, t in enumerate(targets)}
    return class2id

def get_pretrained_emb(emb_path):
    """Pretrained word embeddingsファイルから辞書w2id, 重みWを取得する"""
    lines = open(emb_path).readlines()
    lines = lines[1:]  # 1行目を除く
    w2id = {}
    params = []
    for i, line in enumerate(tqdm(lines)):
        split = line.replace('\n', '').split(' ')
        w2id[split[0]] = i + 1
        params.append(split[1:-1])

    # w2idの作成
    w2id['<UNK>'] = UNK_ID
    w2id['<TARGET>'] = len(w2id)
    # Wの作成
    params.insert(0, [0.0]*len(params[0]))  # <UNK>のパラメータ
    params.append([0.0]*len(params[0]))  # <TARGET>のパラメータ
    W = numpy.array(params, numpy.float32)
    return w2id, W


def make_context_array(words, w2id):
    """文中の単語をIDに変換する"""
    ids = [w2id.get(w, UNK_ID) for w in words]
    return numpy.array(ids, numpy.int32)


def make_target_array(target, class2id):
    """対象単語をIDに変換する"""
    return numpy.array([class2id[target]], numpy.int32)


def split_text(lines, to_kana):
    """左文脈, 右文脈, 対象単語に分割する"""
    left_words, right_words, targets = [], [], []
    for line in tqdm(lines):
        m = re.match(split_regex, line.replace('\n', ''))
        if not m:
            continue
        left_text, target, right_text = m.groups()
        left_words.append(clean_text(left_text, to_kana).split())
        right_words.append(clean_text(right_text, to_kana).split())
        targets.append(target)
    return left_words, right_words, targets


def make_dataset(path_or_data, w2id=None, class2id=None, vocab_size=40000, min_freq=1, n_encoder=2, to_kana=False, emb=None):
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
    left_words, right_words, targets = split_text(lines, to_kana)
    initialW = None

    if not w2id or not class2id:
        if not emb:
            words = [w for words in left_words for w in words] + [w for words in right_words for w in words]
            w2id = get_vocab(words, vocab_size, min_freq)
        else:
            w2id, initialW = get_pretrained_emb(emb)
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
    converters = {'w2id': w2id, 'class2id': class2id, 'initialW': initialW}

    return dataset, converters


def tagging(err, ans):
    """
    :params err: 分かち書きされた誤り文
    :params ans: 分かち書きされた正解文
    (条件: len(err) == len(ans) and err != ans)
    :return test: errとansの不一致箇所の1つをタグ(<>)付けした文
    """
    diff_ids = [i for i in range(len(err)) if err[i] != ans[i]]
    idx = diff_ids[0] if len(diff_ids) == 1 else diff_ids[1]
    test = ans[:idx] + '<' + ans[idx] + '>' + ans[idx+1:]
    return test


def graph(model):
    """モデルのネットワークグラフを描写する"""
    import chainer.computational_graph as c
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
