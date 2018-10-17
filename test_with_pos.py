import json
import argparse
import numpy
import nets
import train_with_pos
from train_with_pos import seq_convert, pos2onehotW
from utils import tagging
from utils_pos import make_dataset_with_pos
import chainer
from chainer.backends import cuda


def main():
    # args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', type=int, default=0, help='Batch size')
    parser.add_argument('--gpuid', type=int, default=-1, help='GPU ID')
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='Type of RNN')
    parser.add_argument('--model_dir', required=True, help='Directory of trained models')
    parser.add_argument('--err', required=True, help='Segmented error text file')
    parser.add_argument('--ans', required=True, help='Segmented answer text file')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch of model to use')
    args = parser.parse_args()

    # prepare
    vocab = json.load(open(args.model_dir + '/vocab.json', 'r'))
    w2id = vocab['w2id']
    id2w = {v: k for k, v in w2id.items()}
    class2id = vocab['class2id']
    id2class = {v: k for k, v in class2id.items()}
    pos2id = vocab['pos2id']
    train_with_pos.pos2onehotW = vocab['pos2onehotW']
    n_vocab = len(w2id)
    n_class = len(class2id)
    opts = json.load(open(args.model_dir + '/opts.json'))
    n_units = opts['unit']
    n_layer = opts['layer']
    dropout = opts['dropout']
    pos_level = opts['pos_level']
    model_file = args.model_dir + '/model-e{}.npz'.format(args.epoch)

    # model
    model = nets.AttnContextClassifierWithPos(n_vocab, n_units, n_class, n_layer, dropout, args.rnn)
    chainer.serializers.load_npz(model_file, model)
    if args.gpuid >= 0:
        cuda.get_device(args.gpuid).use()
        model.to_gpu(args.gpuid)

    # test
    err_data = open(args.err).readlines()
    ans_data = open(args.ans).readlines()
    testdata = [tagging(err, ans) for err, ans in zip(err_data, ans_data)
                if len(err) == len(ans) and err != ans]
    test, _ = make_dataset_with_pos(testdata, pos_level, w2id, class2id, pos2id, pos2onehotW)

    count, t = 0, 0
    if args.batchsize == 0:
        for i in range(len(test)):
            lxs, rxs, ts, lps, rps = seq_convert([test[i]])
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                predict = model.predict(lxs, rxs, lps, rps, argmax=True)
            left_text = ''.join([id2w.get(int(idx), '') for idx in lxs[0]])
            right_text = ''.join([id2w.get(int(idx), '') for idx in rxs[0]])
            target = id2class.get(int(ts[0]))
            predict = id2class.get(int(predict[0]))
            result = True if predict == target else False
            count += 1
            t += 1 if result else 0
            # print('{} [{} {} {}] {}\t{}'.format(left_text, error, predict, target, right_text, result))

    print('\nAccuracy {:.2f}% ({}/{})'.format(t / count * 100, t, count))


if __name__ == '__main__':
    main()
