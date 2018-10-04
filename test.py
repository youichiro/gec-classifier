import json
import argparse
import nets
from train import seq_convert
from utils import make_dataset
import chainer
from chainer.backends import cuda


def main():
    # args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', type=int, default=300, help='Batch size')
    parser.add_argument('--gpuid', type=int, default=-1, help='GPU ID')
    parser.add_argument('--save_dir', required=True, help='Directory to save results')
    parser.add_argument('--test', required=True, help='Test dataset file')
    parser.add_argument('--model', required=True, help='Trained model file')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    # prepare
    vocab = json.load(open(args.save_dir + '/vocab.json', 'r'))
    w2id = vocab['w2id']
    id2w = {v: k for k, v in w2id.items()}
    classes = vocab['classes']
    reversed_classes = {v: k for k, v in classes.items()}
    n_vocab = len(w2id)
    n_class = len(classes)

    opts = json.load(open(args.save_dir + '/opts.json'))
    n_units = opts['unit']
    n_layer = opts['layer']
    dropout = opts['dropout']

    # model
    model = nets.AttnContextClassifier(n_vocab, n_units, n_class, n_layer)
    chainer.serializers.load_npz(args.model, model)
    if args.gpuid >= 0:
        cuda.get_device(args.gpuid).use()
        model.to_gpu(args.gpuid)

    # test
    test, _, _ = make_dataset(args.test, w2id, classes)
    for i in range(0, len(test), args.batchsize):
        lxs, rxs, ts = seq_convert(test[i:i + args.batchsize], args.gpuid)
        predict_classes = model.predict(lxs, rxs, argmax=True)
        for i in range(len(lxs)):
            left_text = ''.join([id2w.get(idx, '') for idx in lxs[i]])
            right_text = ''.join([id2w.get(idx, '') for idx in rxs[i]])
            target = reversed_classes.get(ts[i][0])
            predict = reversed_classes.get(predict_classes[i])
            result = True if predict == target else False
            print('{} {}({}) {} {}'.format(left_text, predict, target, right_text, result))


if __name__ == '__main__':
    main()
