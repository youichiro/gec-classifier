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
        output = checker.correction_test(err, ans)

    acc = checker.acc / checker.total_predict_num * 100
    acc_one = checker.acc_of_one / checker.n_of_one * 100
    n_multi_prediction = checker.total_predict_num - checker.n_of_one
    acc_multi = (checker.acc - checker.acc_of_one) / n_multi_prediction * 100
    print(f"""
    \n[Total]
    Accuracy: {acc:.5}%
    # sentence: {checker.n}
    # total prediction: {checker.total_predict_num}
    \n[For one error]
    Accuracy: {acc_one:.5}%
    # sentence: {checker.n_of_one}
    # total prediction: {checker.n_of_one}
    \n[For multiple error]
    Accuracy: {acc_multi:.5}%
    # sentence: {checker.n - checker.n_of_one}
    # total prediction: {n_multi_prediction}
    """)


if __name__ == '__main__':
    main()

