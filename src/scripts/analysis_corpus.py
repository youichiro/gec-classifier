# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.pardir)

import argparse
from tqdm import tqdm
from mecab import Mecab
from collections import Counter
from utils import TARGETS, TARGET_PARTS, get_target_positions


def convert_pos(pos):
    # if pos[:2] == '助詞':
    #     return '助詞'
    if pos[:2] == '動詞':
        return '動詞'
    if pos[:4] == '補助記号':
        return '補助記号'
    if pos[:2] == '記号':
        return '記号'
    if pos[:3] == '感動詞':
        return '感動詞'
    if pos[:2] == '名詞':
        return '名詞'
    return pos


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
            prev_pos = convert_pos(parts[target_id-1])
            part_pair_list.append(prev_pos)

            out = f'{words[target_id-1]}:{parts[target_id-1]}\t{words[target_id]}:{parts[target_id]}\n'
            f_out.write(out)

    counter = Counter(part_pair_list)
    for k, v in counter.most_common():
        f_dic.write(f'{v}\t{k}\n')

    f_out.close()
    f_dic.close()


if __name__ == '__main__':
    main()
