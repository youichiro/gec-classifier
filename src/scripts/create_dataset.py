# -*- coding: utf-8 -*-
"""
- テキストからランダムに1つの訂正対象を選択してマークする
  - ex) 彼に車を預ける → 彼 に 車 <を> 預ける
- 訂正対象の削除パターンも作りたいので，1単語前が名詞・代名詞・助動詞・助詞・接尾辞-名詞的・動詞かつ今の品詞が助詞・助動詞でない場合，助詞の削除ラベルをつける
  - ex) 彼に車を2台預ける → 彼 に 車 を 2 台 <DEL> 預ける
"""
import sys
import os
sys.path.append(os.pardir)

import random
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import TARGETS, TARGET_PARTS, clean_text, get_target_positions, get_del_positions
from mecab import Mecab, tagger


def make_labeled_sentence(line, args):
    """テキストからラベル付けを行う箇所を決定し，ラベル付けした文を返す"""
    line = clean_text(line)  # クリーニング
    words, parts = tagger(line, args.mecab_dic)
    words, parts = Mecab.preprocessing_to_particle(words, parts, TARGETS, TARGET_PARTS)  # 2単語になった助詞を1単語に変換しておく
    target_idx = get_target_positions(words, parts)  # 助詞の位置を検出
    del_idx = get_del_positions(words, parts)  # 削除ラベルを挿入する位置を検出
    n_target = len(target_idx) + len(del_idx)

    # ラベル付けする位置を決める
    if n_target == 0 or len(words) > args.maxlen:
        return None
    elif len(target_idx) == 0:
        target_id = random.choice(del_idx)
    elif len(del_idx) == 0:
        target_id = random.choice(target_idx)
    else:
        # 文中に複数対象がある場合はランダムに1箇所選ぶ
        # 削除ラベルをX%の確率で作成する
        if random.random() < float(args.del_rate):
            target_id = random.choice(del_idx)
        else:
            target_id = random.choice(target_idx)

    # ラベル付け
    if parts[target_id][:2] == '助詞' or parts[target_id] == '助動詞':
        labeled_sentence = '{} <{}> {}'.format(
            ' '.join(words[:target_id]), words[target_id], ' '.join(words[target_id+1:]))
    else:
        # 削除ラベルの場合
        labeled_sentence = '{} <{}> {}'.format(
            ' '.join(words[:target_id]), 'DEL', ' '.join(words[target_id:]))

    return labeled_sentence


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--corpus', required=True, help='Corpus file')
    parser.add_argument('--save-train', required=True, help='Save file of train data')
    parser.add_argument('--save-valid', required=True, help='Save file of validation data')
    parser.add_argument('--valid-size', type=int, default=1000, help='Size of validation data')
    parser.add_argument('--maxlen', type=int, default=70, help='Max num of words in a sentence')
    parser.add_argument('--mecab-dic', default='/tools/env/lib/mecab/dic/unidic', help='MeCab dict path')
    parser.add_argument('--del-rate', default=0.1, help='Rate of deletion label (0.0~1.0)')
    args = parser.parse_args()

    lines = open(args.corpus, 'r', encoding='utf-8').readlines()
    random.shuffle(lines)  # 順序をシャッフル

    labeled_data = Parallel(n_jobs=-1, verbose=1)([delayed(make_labeled_sentence)(line, args) for line in tqdm(lines)])
    labeled_data = [s for s in labeled_data if s is not None]

    print('Saving ...')
    with open(args.save_valid, 'w') as f:
        for s in labeled_data[:args.valid_size]:
            f.write(s + '\n')

    with open(args.save_train, 'w') as f:
        for s in labeled_data[args.valid_size:]:
            f.write(s + '\n')


if __name__ == '__main__':
    main()

