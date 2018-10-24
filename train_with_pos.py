import os
import json
import numpy
import argparse

import chainer
from chainer.dataset import convert
from chainer.backends import cuda
from chainer.training import extensions
from chainer import functions as F

import nets
from utils import make_dataset, IGNORE_ID, UNK_ID
from utils_pos import make_dataset_with_pos
from train import SaveModel

pos2onehotW = []


def seq_convert(batch, device=None):
    lxs, rxs, ts, lps, rps  = zip(*batch)
    lxs_block = convert.concat_examples(lxs, device, padding=IGNORE_ID)
    rxs_block = convert.concat_examples(rxs, device, padding=IGNORE_ID)
    ts_block = convert.concat_examples(ts, device)

    lps_block = convert.concat_examples(lps, device, padding=len(pos2onehotW)-1)  # (bs, len(seq))
    rps_block = convert.concat_examples(rps, device, padding=len(pos2onehotW)-1)  # (bs, len(seq))

    # (バッチ×品詞ID系列)行列に品詞onehotを埋め込む
    lps_len = [len(lps) for lps in lps_block]
    lps_section = numpy.cumsum(lps_len[:-1])
    lps_concat = numpy.concatenate(lps_block, axis=0)
    lps_onehot = pos2onehotW[lps_concat]
    lps_block = numpy.split(lps_onehot, lps_section, 0)
    lps_block = numpy.array(lps_block)
    lps_block = convert.to_device(device, lps_block)

    rps_len = [len(rps) for rps in rps_block]
    rps_section = numpy.cumsum(rps_len[:-1])
    rps_concat = numpy.concatenate(rps_block, axis=0)
    rps_onehot = pos2onehotW[rps_concat]
    rps_block = numpy.split(rps_onehot, rps_section, 0)
    rps_block = numpy.array(rps_block)
    rps_block = convert.to_device(device, rps_block)

    # #TODO: ここでメモリエラー
    # lps_list = lps_block.tolist()
    # rps_list = rps_block.tolist()

    # for i in range(len(lps_list)):
    #     for j in range(len(lps_list[i])):
    #         lps_list[i][j] = pos2onehotW[lps_list[i][j]]
    # for i in range(len(rps_list)):
    #     for j in range(len(rps_list[i])):
    #         rps_list[i][j] = pos2onehotW[rps_list[i][j]]

    # lps_block = numpy.array(lps_list, numpy.float32)
    # rps_block = numpy.array(rps_list, numpy.float32)

    return (lxs_block, rxs_block, ts_block, lps_block, rps_block)


def unknown_rate(data):
    n_unk = sum((ls == UNK_ID).sum() + (rs == UNK_ID).sum() for ls, rs, _, _, _ in data)
    total = sum(ls.size + rs.size for ls, rs, _, _, _ in data)
    return n_unk / total


def main():
    # args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vocabsize', type=int, default=40000, help='Max size of vocablary')
    parser.add_argument('--minfreq', type=int, default=1, help='Min token frequency')
    parser.add_argument('--epoch', type=int, default=30, help='Max epochs')
    parser.add_argument('--batchsize', type=int, default=300, help='Batch size')
    parser.add_argument('--gpuid', type=int, default=-1, help='GPU ID')
    parser.add_argument('--unit', type=int, default=300, help='Number of hidden layer units')
    parser.add_argument('--layer', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='Type of RNN')
    parser.add_argument('--train', required=True, help='Train dataset file')
    parser.add_argument('--valid', required=True, help='Validation dataset file')
    parser.add_argument('--save_dir', required=True, help='Directory to save results')
    parser.add_argument('--pos_level', type=int, required=True, choices=[1, 2, 3], help=(
                             'Level of POS: 1 means major, 2 means medium and 3 means small'))
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    # prepare
    train, converters = make_dataset_with_pos(args.train, args.pos_level,
                                              vocab_size=args.vocabsize, min_freq=args.minfreq)
    w2id, class2id = converters['w2id'], converters['class2id']
    global pos2onehotW
    pos2id, pos2onehotW = converters['pos2id'], converters['pos2onehotW']
    valid, _ = make_dataset_with_pos(args.valid, args.pos_level, w2id, class2id, pos2id, pos2onehotW)
    n_vocab = len(w2id)
    n_class = len(class2id)
    unk_rate = unknown_rate(train)
    vocab = {'class2id': class2id, 'w2id': w2id, 'pos2id': pos2id, 'pos2onehotW': pos2onehotW.tolist()}
    args.__dict__['pos_level'] = args.pos_level
    args.__dict__['train_size'] = len(train)
    args.__dict__['unknown_rate'] = unk_rate
    os.makedirs(args.save_dir, exist_ok=True)
    json.dump(vocab, open(args.save_dir + '/vocab.json', 'w'), ensure_ascii=False)
    json.dump(args.__dict__, open(args.save_dir + '/opts.json', 'w'))
    print('Train size:', len(train))
    print('Vocab size:', n_vocab)
    print('Unknown rate: {:.2f}%'.format(unk_rate * 100))

    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, batch_size=args.batchsize,
                                                  repeat=False, shuffle=False)

    # model
    model = nets.AttnContextClassifierWithPos(n_vocab, args.unit, n_class, args.layer, args.dropout, args.rnn)
    if args.gpuid >= 0:
        cuda.get_device_from_id(args.gpuid).use()
        model.to_gpu(args.gpuid)

    # trainer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    updater = chainer.training.StandardUpdater(train_iter, optimizer,
                                               converter=seq_convert, device=args.gpuid)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.save_dir)
    trainer.extend(extensions.Evaluator(valid_iter, model, converter=seq_convert, device=args.gpuid))
    trainer.extend(SaveModel(model, args.save_dir))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy',
        'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    # train
    trainer.run()


if __name__ == '__main__':
    main()
