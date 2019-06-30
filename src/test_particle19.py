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
    parser.add_argument('--lm-data', default=False, help='Language model data')
    parser.add_argument('--reverse', default=False, action='store_true',
                        help='Whether to reverse the prediction order')
    parser.add_argument('--threshold', type=float, default=0.0, help='Threshold of correction [0.0-1.0]')
    parser.add_argument('--save-dir', required=True, help='Save directory')
    args = parser.parse_args()

    mecab_dict_file = os.environ['MECABDIC']
    model_file = args.model_dir + f'/model-e{args.epoch}.npz'
    vocab_file = args.model_dir + '/vocab.json'
    opts_file = args.model_dir + '/opts.json'
    checker = Checker(mecab_dict_file, model_file, vocab_file, opts_file, args.lm_data, args.reverse, args.threshold)

    error_data = open(args.err).readlines()
    answer_data = open(args.ans).readlines()

    save_dir = args.save_dir + '/' + args.model_dir.split('/')[-1]
    os.makedirs(save_dir, exist_ok=True)

    if args.threshold > 0.0:
        save_dir += f'/threshold{args.threshold}'
        os.makedirs(save_dir, exist_ok=True)

    save_file = save_dir + '/model.hyp' if not args.lm_data else save_dir + '/model+lm.hyp'
    save_file_char = save_dir + '/model_char.hyp' if not args.lm_data else save_dir + '/model+lm_char.hyp'
    save_file_out = save_dir + '/model.out' if not args.lm_data else save_dir + '/model+lm.out'
    save_file_m2score = save_dir + '/m2score.txt' if not args.lm_data else save_dir + '/m2score_model+lm.txt'

    f_hyp = open(save_file, 'w')
    f_hyp_char = open(save_file_char, 'w')
    f_hyp_out = open(save_file_out, 'w')
    same_count = 0

    for i, (err, ans) in enumerate(zip(tqdm(error_data), answer_data)):
        err = err.replace('\n', '')
        ans = ans.replace('\n', '')
        hyp = checker.correction(err)
        is_same = hyp == ans

        f_hyp.write(hyp + '\n')
        f_hyp_char.write(' '.join(hyp) + '\n')
        f_hyp_out.write(f'--- {i} ---\nerr: {err}\nhyp: {hyp}\nans: {ans}\n{is_same}\n\n')

        if is_same:
            same_count += 1

    print('Saved to ' + save_file)
    print('Saved to ' + save_file_char)
    print('Saved to ' + save_file_out)
    f_hyp.close()
    f_hyp_char.close()
    f_hyp_out.close()

    # m2scoreを実行
    cmd = f"python /lab/ogawa/tools/m2scorer/m2scorer {save_file_char} {args.save_dir}/naist_test19_char.m2 > {save_file_m2score}"
    subprocess.call(cmd, shell=True)
    print(f'\nSaved to {save_file_m2score}')
    print(f'Same rate: {same_count / len(answer_data) * 100:.4}% ({same_count}/{len(answer_data)})')


if __name__ == '__main__':
    main()

