import sys
import os
sys.path.append(os.pardir)

import random
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import TARGETS, TARGET_PARTS, clean_text, get_target_positions, get_del_positions
from mecab import Mecab, tagger


corpus = '../datasets/v2/bccwj+lang8/bccwj+lang8.txt'


def count(line):
    line = clean_text(line)  # クリーニング
    words, parts = tagger(line, args.mecab_dic)
    words, parts = Mecab.preprocessing_to_particle(words, parts, TARGETS, TARGET_PARTS)
    target_idx = get_target_positions(words, parts)
    del_idx = get_del_positions(words, parts)
    return len(target_idx), len(del_idx)


corpus = open(corpus, 'r', encoding='utf-8').readlines()
data = Parallel(n_jobs=-1, verbose=1)([delayed(count)(line) for line in tqdm(corpus)])
sum_targets = sum([d[0] for d in data])
sum_dels = sum([d[1] for d in data])

print('sum_targets: ' + str(sum_targets))
print('sum_dels: ' + str(sum_dels))

