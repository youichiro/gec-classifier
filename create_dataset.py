"""
テキストからランダムに1つの格助詞を選択してマークする
ex) 彼に車を預ける → 彼 に 車 <を> 預ける
"""
import os
import random
import argparse
from tqdm import tqdm
import MeCab


mecab_dict_path = '/tools/env/lib/mecab/dic/unidic'
TARGETS = ['が', 'を', 'に', 'で']
TARGET_PART = '助詞-格助詞'


class Mecab:
    def __init__(self, dict_path):
        self.t = MeCab.Tagger('-d {}'.format(dict_path))

    def tagger(self, text):
        n = self.t.parse(text)
        lines = n.split('\n')
        words, parts = [], []
        for line in lines[:-2]:
            words.append(line.split('\t')[0])
            parts.append(line.split('\t')[4])
        return words, parts


def get_target_positions(words, parts):
    target_idx = [i for i, (w, p) in enumerate(zip(words, parts))
                  if p == TARGET_PART and w in TARGETS \
                  and i != 0 and i != len(words) - 1]  # 文頭と文末の格助詞は除く
    return target_idx


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--corpus', required=True, help='Corpus file')
    parser.add_argument('--save_train', required=True, help='Save file of train data')
    parser.add_argument('--save_valid', required=True, help='Save file of validation data')
    parser.add_argument('--valid_size', type=int, default=1000, help='Size of validation data')
    parser.add_argument('--maxlen', type=int, default=70, help='Max num of words in a sentence')
    args = parser.parse_args()

    mecab = Mecab(mecab_dict_path)
    count = 0
    if os.path.exists(args.save_train):
        os.remove(args.save_train)
    if os.path.exists(args.save_valid):
        os.remove(args.save_valid)

    lines = open(args.corpus, 'r', encoding='utf-8').readlines()
    for line in tqdm(lines):
        words, parts = mecab.tagger(line.rstrip())
        target_idx = get_target_positions(words, parts)
        n_target = len(target_idx)
        if not n_target or len(words) > args.maxlen:
            continue
        elif n_target == 1:
            target_id = target_idx[0]
        else:
            target_id = random.choice(target_idx)
        marked_sentence = '{} <{}> {}'.format(
            ' '.join(words[:target_id]), words[target_id], ' '.join(words[target_id+1:]))

        save_path = args.save_valid if count < args.valid_size else args.save_train
        open(save_path, 'a').write(marked_sentence + '\n')
        count += 1


if __name__ == '__main__':
    main()

