# -*- coding: utf-8 -*-
import os
import argparse
import subprocess
from tqdm import tqdm
from correction_particle19 import Checker


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-dir', required=True, help='Model dir')
    parser.add_argument('--epoch', required=True, help='Model epoch')
    parser.add_argument('--err', required=True, help='Error corpus')
    parser.add_argument('--ans', required=True, help='Answer corpus')
    parser.add_argument('--reverse', default=False, action='store_true',
                        help='Whether to reverse the prediction order')
    parser.add_argument('--show', default=False, action='store_true', help='Whether to show results')
    parser.add_argument('--save-dir', default='results/v2', help='Save directory')
    args = parser.parse_args()

    mecab_dict_file = os.environ['MECABDIC']
    model_file = args.model_dir + f'/model-e{args.epoch}.npz'
    vocab_file = args.model_dir + '/vocab.json'
    opts_file = args.model_dir + '/opts.json'
    checker = Checker(mecab_dict_file, model_file, vocab_file, opts_file, args.reverse)

    error_data = open(args.err).readlines()
    answer_data = open(args.ans).readlines()

    if args.save_dir:
        save_dir = args.save_dir + '/' + args.model_dir.split('/')[-1]
        os.makedirs(save_dir, exist_ok=True)
        save_file = save_dir + '/model.hyp'
        f_hyp = open(save_file, 'w')
        save_file_char = save_dir + '/model_char.hyp'
        f_hyp_char = open(save_file_char, 'w')

    if not args.show:
        error_data = tqdm(error_data)

    for i, (err, ans) in enumerate(zip(error_data, answer_data)):
        err = err.replace('\n', '')
        ans = ans.replace('\n', '')
        hyp = checker.correction(err)

        if args.show:
            print(f'--- {i} ---')
            print(f'err: {err}')
            print(f'hyp: {hyp}')
            print(f'ans: {ans}')
            print()

        if args.save_dir:
            f_hyp.write(hyp + '\n')
            f_hyp_char.write(' '.join(hyp) + '\n')

    if args.save_dir:
        print('Saved to ' + save_file)
        print('Saved to ' + save_file_char)
        f_hyp.close()
        f_hyp_char.close()

    # m2scoreを実行
    cmd = f"python /lab/ogawa/tools/m2scorer/m2scorer {save_file_char} {args.save_dir}/naist_test19_char.m2 > {save_dir}/m2score.txt"
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    main()

