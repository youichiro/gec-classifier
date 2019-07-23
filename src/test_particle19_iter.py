# -*- coding: utf-8 -*-
import os
import argparse
import subprocess
from tqdm import tqdm
from correction_particle19 import Checker


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-dir', required=True, help='Model dir')
    parser.add_argument('--epoch', required=True, help='Model epoch')
    parser.add_argument('--reverse', default=False, action='store_true',
                        help='Whether to reverse the prediction order')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Threshold of correction [0.0-1.0]')
    args = parser.parse_args()

    mecab_dict_file = os.environ['MECABDIC']
    model_file = args.model_dir + f'/model-e{args.epoch}.npz'
    vocab_file = args.model_dir + '/vocab.json'
    opts_file = args.model_dir + '/opts.json'
    checker = Checker(mecab_dict_file, model_file, vocab_file,
                      opts_file, reverse=args.reverse, threshold=args.threshold)

    text = ''
    while text != 'end':
        text = input()
        res = checker.correction(text)
        print(res, end='\n\n')


if __name__ == '__main__':
    main()
