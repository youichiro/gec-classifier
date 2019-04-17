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

    # precision = checker.precision / checker.total_predict_num * 100
    # recall = checker.recall / checker.total_error_num * 100
    # print(f"""
    # \n[Total]
    # Precision: {precision:.5}% ({checker.precision}/{checker.total_predict_num})
    # Recall: {recall:.5}% ({checker.recall}/{checker.total_error_num})
    # # sentence: {checker.num_sentence}
    # # error: {checker.error}
    # """)
    print(f'# sentence: {checker.num_sentence}')
    print(f'# error sentence: {checker.error}')
    print(f'# total prediction: {checker.total_predict_num}')
    print(f'# total error: {checker.total_error_num}')
    print(f'# accurate prediction: {checker.accurate}')
    print(f'# accurate prediction of errors: {checker.accurate_of_error}')
    print(f'# accurate prediction of non errors: {checker.total_predict_num - checker.accurate_of_error}')
    print(f'Accuracy: {checker.accurate / checker.total_predict_num * 100:.2f}%')
    print(f'Accuracy of errors: {checker.accurate_of_error / checker.total_error_num * 100:.2f}%')
    print(f'Accuracy of non errors: {(checker.accurate - checker.accurate_of_error) / (checker.total_predict_num - checker.total_error_num) * 100:.2f}%')
    print()

    error_num_dic = {}
    for _, error_num in checker.target_statistic:
        if error_num in error_num_dic.keys():
            error_num_dic[error_num] += 1
        else:
            error_num_dic[error_num] = 1
    print('[Num of error in a sentence]')
    for k, v in sorted(error_num_dic.items()):
        print(f'{k}: {v}', end='  ')
    print()

    print('\n[confusion matrix (error -> answer)]')
    print_confusion_matrix(checker.confusion_target_to_answer)

    print('\n[confusion matrix (error -> predict)]')
    print_confusion_matrix(checker.confusion_target_to_predict)

    print('\n[confusion matrix (predict -> answer)]')
    print_confusion_matrix(checker.confusion_predict_to_answer)


def print_confusion_matrix(dic):
    label = ['が', 'を', 'に', 'で']
    sum_rows = {'が': 0, 'を': 0, 'に': 0, 'で': 0}
    for k, v in dic.items():
        sum_rows[k[0]] += v

    # print('\t' + '\t'.join(label))
    # print('--|' + '----'*10)
    # for i in range(len(label)):
    #     print(label[i] + '|', end='\t')
    #     for j in range(len(label)):
    #         try:
    #             print(dic[f'{label[i]}->{label[j]}'], end='\t')
    #         except KeyError:
    #             print('0', end='\t')
    #     print()
    # print()

    print('\t' + '\t'.join(label) + '\t(total row num)')
    print('--|' + '----'*15)
    for i in range(len(label)):
        print(label[i] + '|', end='\t')
        for j in range(len(label)):
            try:
                print(f'{dic[f"{label[i]}->{label[j]}"] / sum_rows[label[i]] * 100 :.2f}%', end='\t')
            except KeyError:
                print('0', end='\t')
        print(sum_rows[label[i]])


if __name__ == '__main__':
    main()

