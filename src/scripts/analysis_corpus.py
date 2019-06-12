# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.pardir)

import argparse
from tqdm import tqdm
from mecab import Mecab
from collections import Counter

TARGETS = ['が', 'の', 'を', 'に', 'へ', 'と', 'より', 'から', 'で', 'や',
           'は', 'には', 'からは', 'とは', 'では', 'へは', 'までは', 'よりは', 'まで', '']  # 19種類+削除
TARGET_PARTS = ['助詞-格助詞', '助詞-副助詞', '助詞-係助詞', '助詞-接続助詞',
                '助詞-終助詞', '助詞-準体助詞', '助詞']  # '助詞'はオリジナル設定


def get_target_positions(words, parts):
    """訂正対象箇所のインデックスを返す"""
    target_idx = [i for i, (w, p) in enumerate(zip(words, parts))
                  if p in TARGET_PARTS and w in TARGETS
                  and i != 0 and i != len(words) - 1]  # 文頭と文末の助詞は除く
    return target_idx


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--corpus', required=True, help='corpus')
    parser.add_argument('--mecab-dic', default='/tools/env/lib/mecab/dic/unidic', help='mecab-dic')
    parser.add_argument('--save-dir', required=True, help='save directory')
    args = parser.parse_args()

    mecab = Mecab(args.mecab_dic)
    corpus = open(args.corpus).readlines()
    part_pair_list = []
    f_out = open(args.save_dir + '/target_and_prev_list.txt', 'w')
    f_dic = open(args.save_dir + '/target_pos_and_prev_pos_dic.txt', 'w')

    for line in tqdm(corpus):
        line = line.replace('\n', '')
        words, parts = mecab.tagger(line)
        target_idx = get_target_positions(words, parts)

        for target_id in target_idx:
            part_pair_list.append(f'{parts[target_id-1]}\t{parts[target_id]}')
            out = f'{words[target_id-1]}:{parts[target_id-1]}\t{words[target_id]}:{parts[target_id]}\n'
            f_out.write(out)

    counter = Counter(part_pair_list)
    for k, v in counter.most_common():
        f_dic.write(f'{v}\t{k}\n')

    f_out.close()
    f_dic.close()


if __name__ == '__main__':
    main()
