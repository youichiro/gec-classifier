# -*- coding: utf-8 -*-
import os
import argparse
from correction import Checker

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, help='Model file (model-eXX.npz)')
    parser.add_argument('--vocab', required=True, help='Vocab file (vocab.json)')
    parser.add_argument('--opts', required=True, help='Opts file (opts.json)')
    parser.add_argument('--err', required=True, help='Error sentence file')
    parser.add_argument('--ans', required=True, help='Answer sentence file')
    parser.add_argument('--forward', default=False, action='store_true',
                        help='Whether to reverse the prefiction order')
    parser.add_argument('--show', default=False, action='store_true',
                        help='Whether to show results')
    args = parser.parse_args()

    mecab_dict_file = os.environ['MECABDIC']
    checker = Checker(mecab_dict_file, args.model, args.vocab, args.opts, not args.forward, args.show)

    error_data = open(args.err, 'r').readlines()
    answer_data = open(args.ans, 'r').readlines()

    for err, ans in zip(error_data, answer_data):
        err = err.replace('\n', '')
        ans = ans.replace('\n', '')
        _ = checker.correction_test(err, ans)

    precision = checker.precision / checker.total_predict_num * 100
    recall = checker.recall / checker.total_error_num * 100
    print(f"""
    \n[Total]
    Precision: {precision:.5}% ({checker.precision}/{checker.total_predict_num})
    Recall: {recall:.5}% ({checker.recall}/{checker.total_error_num})
    # sentence: {checker.num_sentence}
    # error: {checker.error}
    """)

    error_num_dic = {}
    for _, error_num in checker.target_statistic:
        if error_num in error_num_dic.keys():
            error_num_dic[error_num] += 1
        else:
            error_num_dic[error_num] = 1
    print('[# error VS # sentence]')
    for k, v in sorted(error_num_dic.items()):
        print(f'\t{k}: {v}')

    print('\n[NAIST confusion matrix (error -> answer)]')
    for k, v in sorted(checker.naist_confusion.items()):
        print(f'\t{k}: {v}')

    print('\n[Predict confusion matrix (error -> predict)]')
    for k, v in sorted(checker.predict_confusion.items()):
        print(f'\t{k}: {v}')

if __name__ == '__main__':
    main()

