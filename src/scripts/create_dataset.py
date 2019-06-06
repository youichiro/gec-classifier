# -*- coding: utf-8 -*-
"""
テキストからランダムに1つの格助詞を選択してマークする
ex) 彼に車を預ける → 彼 に 車 <を> 預ける
"""
import sys
import os
sys.path.append(os.pardir)

import random
import argparse
from tqdm import tqdm
import MeCab
from utils import clean_text
from mecab import Mecab


# TARGETS = ['が', 'を', 'に', 'で']
# TARGET_PART = '助詞-格助詞'

TARGETS = ['が', 'の', 'を', 'に', 'へ', 'と', 'より', 'から', 'で', 'や',
           'は', 'には', 'からは', 'とは', 'では', 'へは', 'までは', 'よりは', 'まで', '']  # 19種類+削除
TARGET_PARTS = ['助詞-格助詞', '助詞-副助詞', '助詞-係助詞', '助詞-接続助詞', '助詞-終助詞', '助詞-準体助詞', '助詞']  # '助詞'はオリジナル設定


def get_target_positions(words, parts):
    """訂正対象箇所のインデックスを返す"""
    target_idx = [i for i, (w, p) in enumerate(zip(words, parts))
                  if p in TARGET_PARTS and w in TARGETS \
                  and i != 0 and i != len(words) - 1]  # 文頭と文末の助詞は除く
    return target_idx


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--corpus', required=True, help='Corpus file')
    parser.add_argument('--save-train', required=True, help='Save file of train data')
    parser.add_argument('--save-valid', required=True, help='Save file of validation data')
    parser.add_argument('--valid-size', type=int, default=1000, help='Size of validation data')
    parser.add_argument('--maxlen', type=int, default=70, help='Max num of words in a sentence')
    parser.add_argument('--mecab-dic', default='/tools/env/lib/mecab/dic/unidic', help='MeCab dict path')
    args = parser.parse_args()

    mecab = Mecab(args.mecab_dic)
    count = 0
    if os.path.exists(args.save_train):
        os.remove(args.save_train)
    if os.path.exists(args.save_valid):
        os.remove(args.save_valid)

    lines = open(args.corpus, 'r', encoding='utf-8').readlines()
    random.shuffle(lines)  # 順序をシャッフル
    for line in tqdm(lines):
        line = clean_text(line.rstrip())  # 全角→半角，数字→#
        words, parts = mecab.tagger(line)  # 形態素解析
        words, parts = mecab.preprocessing_to_particle(words, parts, TARGETS, TARGET_PARTS)  # 2単語になった助詞を1単語に変換しておく
        target_idx = get_target_positions(words, parts)  # 助詞の位置を検出
        n_target = len(target_idx)
        if not n_target or len(words) > args.maxlen:
            continue
        elif n_target == 1:
            target_id = target_idx[0]
        else:
            # 文中に複数対象がある場合はランダムに1箇所選ぶ
            target_id = random.choice(target_idx)
        marked_sentence = '{} <{}> {}'.format(
            ' '.join(words[:target_id]), words[target_id], ' '.join(words[target_id+1:]))

        save_path = args.save_valid if count < args.valid_size else args.save_train
        open(save_path, 'a').write(marked_sentence + '\n')
        count += 1


if __name__ == '__main__':
    main()

