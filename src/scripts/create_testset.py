# -*- coding: utf-8 -*-
import argparse
from utils import tagging


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--err', required=True, help='Segmented error text file')
    parser.add_argument('--ans', required=True, help='Segmented answer text file')
    parser.add_argument('--save', required=True, help='Save file of testdata')
    args = parser.parse_args()

    err_data = open(args.err).readlines()
    ans_data = open(args.ans).readlines()
    testdata = [tagging(err, ans) for err, ans in zip(err_data, ans_data)
                if len(err) == len(ans) and err != ans]
    testdata = [test for test in testdata
                if '<が>' in test or '<を>' in test or '<に>' in test or '<で>' in test]
    with open(args.save, 'w') as f:
        for test in testdata:
            f.write(test)


if __name__ == '__main__':
    main()
