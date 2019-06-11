# -*- coding: utf-8 -*-
import re
import numpy
import random
import neologdn
import emoji
import regex
from tqdm import tqdm
from collections import Counter
from pykakasi import kakasi
from joblib import Parallel, delayed


IGNORE_ID = -1
UNK_ID = 0
split_regex = r'<[がのをにへとよりからでやはにま]{1,4}>'
split_regex = re.compile(split_regex)
kakasi = kakasi()
kakasi.setMode('J', 'H')  # J(漢字) -> H(ひらがな)
conv = kakasi.getConverter()


def preprocess_text(text, to_kana=False):
    """クリーニング，かな変換を行う"""
    text = clean_text(text)
    if to_kana:
        text = conv.do(text)  # ひらがなに変換
    return text


def clean_text(text):
    """テキストのクリーニング"""
    # 全角スペース\u3000をスペースに変換
    text.replace('\u3000', ' ')
    # 全角→半角，重ね表現の除去
    text = neologdn.normalize(text, repeat=3)
    # 絵文字を削除
    text = ''.join(['' if c in emoji.UNICODE_EMOJI else c for c in text])
    # 桁区切りの除去と数字の置換
    text = re.sub(r'(\d)([,.])(\d+)', r'\1\3', text)
    text = re.sub(r'\d+', '0', text)
    # 半角記号の置換
    text = re.sub(r'[!-/:-@[-`{-~]', r' ', text)
    # 全角記号の置換 (ここでは0x25A0 - 0x266Fのブロックのみを除去)
    text = re.sub(u'[■-♯]', ' ', text)
    # 文頭の「数字列+スペース」を削除
    text = regex.sub(r'^(\p{Nd}+\p{Zs})(.*)$', r'\2', text)
    # 文頭行末の空白は削除
    text = text.strip()
    # 複数の空白を1つにまとめる
    text = re.sub(r'\s+', ' ', text)
    return text


def convert_to_kana(text):
    """平仮名に変換する"""
    return conv.do(text)


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


def get_pretrained_emb(emb_path, vocab_size, to_kana):
    """Pretrained word embeddingsファイルから辞書w2id, 重みWを取得する"""
    lines = open(emb_path).readlines()
    lines = lines[1:vocab_size+1]  # 1行目を除く
    w2id = {}
    params = []
    n = 1
    for i, line in enumerate(lines):
        split = line.replace('\n', '').split(' ')
        word = split[0]
        vec = split[1:-1]
        if word == '</s>': continue
        if to_kana:
            word_kana = conv.do(word)  # カタカナに変換
            if word_kana in w2id.keys():  # 同じ単語があれば先に登録した方のみ保持する
                continue
            w2id[word_kana] = n
            params.append(vec)
        else:
            w2id[word] = n
            params.append(vec)
        n += 1

    # w2idの作成
    # TODO: "よりは" = "より" + "は" のようにベクトルを作る
    w2id['<UNK>'] = UNK_ID
    w2id['<TARGET>'] = len(w2id)
    # Wの作成
    params.insert(0, [0.0]*len(params[0]))  # <UNK>のパラメータ
    params.append([0.0]*len(params[0]))  # <TARGET>のパラメータ
    W = numpy.array(params, numpy.float32)
    # Wの行数とw2idに登録された単語数が一致することを保証
    assert W.shape[0] == len(w2id)
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
        line = line.replace('\n', '')
        search = split_regex.search(line)
        if search is None:
            continue
        target = search.group()[1:-1]
        right_text = line[:search.start()]
        left_text = line[search.end():]

        left_words.append(preprocess_text(left_text, to_kana).split())
        right_words.append(preprocess_text(right_text, to_kana).split())
        targets.append(target)
    return left_words, right_words, targets


def split_text2(line, to_kana):
    line = line.replace('\n', '')
    search = split_regex.search(line)
    if search is None:
        return None, None, None
    target = search.group()[1:-1]
    left_text = line[:search.start()]
    right_text = line[search.end():]
    return left_text, target, right_text


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
    splited_lines = Parallel(n_jobs=-1)([delayed(split_text2)(line, to_kana) for line in tqdm(lines)])
    left_words = [preprocess_text(line[0], to_kana).split() for line in tqdm(splited_lines) if line[0] is not None]
    targets = [preprocess_text(line[1], to_kana).split() for line in tdqm(splited_lines) if line[1] is not None]
    right_words = [preprocess_text(line[2], to_kana).split() for line in tqdm(splited_lines) if line[2] is not None]
    print(left_words)
    # left_words, right_words, targets = split_text(lines, to_kana)
    initialW = None

    if not w2id or not class2id:
        if not emb:
            words = [w for words in left_words for w in words] + [w for words in right_words for w in words]
            w2id = get_vocab(words, vocab_size, min_freq)
        else:
            w2id, initialW = get_pretrained_emb(emb, vocab_size, to_kana)
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
