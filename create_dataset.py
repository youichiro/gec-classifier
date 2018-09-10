"""
テキストからランダムに1つの格助詞を選択してマークする
ex) 彼に車を預ける → 彼 に 車 <を> 預ける
"""
import os
import MeCab
from tqdm import tqdm
import random


data_path = 'datasets/mai2000.1k.txt'
save_train_path = 'datasets/train.txt'
save_dev_path = 'datasets/dev.txt'
mecab_dict_path = '/usr/local/mecab/lib/mecab/dic/unidic'
TARGETS = ['が', 'を', 'に', 'で']
TARGET_PART = '助詞-格助詞'
valid_size = 100
max_len = 70


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
                  if p == TARGET_PART and w in TARGETS]
    return target_idx


def main():
    mecab = Mecab(mecab_dict_path)
    count = 0
    if os.path.exists(save_train_path):
        os.remove(save_train_path)
    if os.path.exists(save_dev_path):
        os.remove(save_dev_path)

    lines = open(data_path, 'r', encoding='utf-8').readlines()
    for line in tqdm(lines):
        words, parts = mecab.tagger(line.rstrip())
        target_idx = get_target_positions(words, parts)
        n_target = len(target_idx)
        if not n_target or len(words) > max_len:
            continue
        elif n_target == 1:
            target_id = target_idx[0]
        else:
            target_id = random.choice(target_idx)
        marked_sentence = '{} <{}> {}'.format(
            ' '.join(words[:target_id]), words[target_id], ' '.join(words[target_id+1:]))
        
        save_path = save_dev_path if count < valid_size else save_train_path
        open(save_path, 'a').write(marked_sentence + '\n')
        count += 1


if __name__ == '__main__':
    main()
