# -*- coding: utf-8 -*-
import os
import argparse
from correction_particle19 import Checker


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-dir', required=True, help='Model dir')
    parser.add_argument('--epoch', required=True, help='Model epoch')
    parser.add_argument('--err', required=True, help='Error corpus')
    parser.add_argument('--ans', required=True, help='Answer corpus')
    parser.add_argument('--reverse', default=False, action='store_true',
                        help='Whether to reverse the prediction order')
    args = parser.parse_args()

    mecab_dict_file = os.environ['MECABDIC']
    model_file = args.model_dir + f'/model-e{args.epoch}.npz'
    vocab_file = args.model_dir + '/vocab.json'
    opts_file = args.model_dir + '/opts.json'
    checker = Checker(mecab_dict_file, model_file, vocab_file, opts_file, args.reverse)

    error_data = open(args.err).readlines()
    answer_data = open(args.ans).readlines()

    for i, (err, ans) in enumerate(zip(error_data, answer_data)):
        err = err.replace('\n', '')
        ans = ans.replace('\n', '')
        hyp = checker.correction(err)

        print(f'--- {i} ---')
        print(f'err: {err}')
        print(f'hyp: {hyp}')
        print(f'ans: {ans}')
        print()


if __name__ == '__main__':
    main()

