import json
import argparse
import nets
from train import seq_convert
from utils import make_dataset, tagging
import chainer
from chainer.backends import cuda


class Eval:
    def __init__(self):
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def count(self, error, predict, answer):
        if error == predict == answer: self.tn += 1
        elif error == predict != answer: self.fn += 1
        elif error != predict == answer: self.tp += 1
        elif error != predict != answer: self.fp += 1

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def f(self):
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())


def main():
    # args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', type=int, default=0, help='Batch size')
    parser.add_argument('--gpuid', type=int, default=-1, help='GPU ID')
    parser.add_argument('--attn', required=True, choices=['disuse', 'global'], help='Type of attention mechanism')
    parser.add_argument('--rnn', required=True, choices=['LSTM', 'GRU'], help='Type of RNN')
    parser.add_argument('--model_dir', required=True, help='Directory of trained models')
    parser.add_argument('--err', required=True, help='Segmented error text file')
    parser.add_argument('--ans', required=True, help='Segmented answer text file')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch of model to use')
    args = parser.parse_args()

    # prepare
    vocab = json.load(open(args.model_dir + '/vocab.json', 'r'))
    w2id = vocab['w2id']
    id2w = {v: k for k, v in w2id.items()}
    class2id = vocab['classes']
    # class2id = vocab['class2id']
    id2class = {v: k for k, v in class2id.items()}
    n_vocab = len(w2id)
    n_class = len(class2id)
    opts = json.load(open(args.model_dir + '/opts.json'))
    n_units = opts['unit']
    n_layer = opts['layer']
    dropout = opts['dropout']
    model_file = args.model_dir + '/model-e{}.npz'.format(args.epoch)

    # model
    if args.attn == 'disuse':
        model = nets.ContextClassifier(n_vocab, n_units, n_class, n_layer, dropout, args.rnn)
    elif args.attn == 'global':
        model = nets.AttnContextClassifier(n_vocab, n_units, n_class, n_layer, dropout, args.rnn)
    chainer.serializers.load_npz(model_file, model)
    if args.gpuid >= 0:
        cuda.get_device(args.gpuid).use()
        model.to_gpu(args.gpuid)

    # test
    err_data = open(args.err).readlines()
    ans_data = open(args.ans).readlines()
    testdata = [tagging(err, ans) for err, ans in zip(err_data, ans_data)
                if len(err) == len(ans) and err != ans]
    test, _, _ = make_dataset([data[0] for data in testdata], w2id, class2id)

    count, t = 0, 0
    evl = Eval()
    if args.batchsize == 0:
        for i in range(len(test)):
            lxs, rxs, ts = seq_convert([test[i]])
            predict = model.predict(lxs, rxs, argmax=True)
            left_text = ''.join([id2w.get(int(idx), '') for idx in lxs[0]])
            right_text = ''.join([id2w.get(int(idx), '') for idx in rxs[0]])
            target = id2class.get(int(ts[0]))
            predict = id2class.get(int(predict[0]))
            error = testdata[i][1]
            result = True if predict == target else False
            # print('{} [{} {} {}] {}\t{}'.format(left_text, error, predict, target, right_text, result))
            count += 1
            t += 1 if result else 0
            evl.count(error, predict, target)
    else:
        for i in range(0, len(test), args.batchsize):
            lxs, rxs, ts = seq_convert(test[i:i + args.batchsize], args.gpuid)
            predict_classes = model.predict(lxs, rxs, argmax=True)
            for j in range(len(lxs)):
                left_text = ''.join([id2w.get(int(idx), '') for idx in lxs[j]])
                right_text = ''.join([id2w.get(int(idx), '') for idx in rxs[j]])
                target = id2class.get(int(ts[j][0]))
                predict = id2class.get(int(predict_classes[j]))
                result = True if predict == target else False
                count += 1
                t += 1 if result else 0

    print('\nAccuracy {:.2f}% ({}/{})'.format(t / count * 100, t, count))
    print('Precision {:.2f}%, Recall {:.2f}%, F {:.2f}%\n(TP:{} FP:{} FN:{} TN:{})'.format(
        evl.precision() * 100, evl.recall() * 100, evl.f() * 100, evl.tp, evl.fp, evl.fn, evl.tn))


if __name__ == '__main__':
    main()

