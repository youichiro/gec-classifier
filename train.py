import os
import json
import numpy
import argparse

import chainer
from chainer.dataset import convert
from chainer.backends import cuda
from chainer.training import extensions

import nets
from utils import make_dataset, get_pretrained_emb, IGNORE_ID, UNK_ID


class SaveModel(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir

    def __call__(self, trainer):
        model_name = 'model-e{}.npz'.format(trainer.updater.epoch)
        chainer.serializers.save_npz(self.save_dir + '/' + model_name, self.model)


def seq_convert(batch, device=None):
    if len(batch[0]) == 3:
        lxs, rxs, ts = zip(*batch)
        lxs_block = convert.concat_examples(lxs, device, padding=IGNORE_ID)
        rxs_block = convert.concat_examples(rxs, device, padding=IGNORE_ID)
        ts_block = convert.concat_examples(ts, device)
        return (lxs_block, rxs_block, ts_block)
    elif len(batch[0]) == 2:
        xs, ts = zip(*batch)
        xs_block = convert.concat_examples(xs, device, padding=IGNORE_ID)
        ts_block = convert.concat_examples(ts, device, padding=IGNORE_ID)
        return (xs_block, ts_block)


def unknown_rate(data):
    if len(data[0]) == 3:
        n_unk = sum((ls == UNK_ID).sum() + (rs == UNK_ID).sum() for ls, rs, ts in data)
        total = sum(ls.size + rs.size for ls, rs, ts in data)
    elif len(data[0]) == 2:
        n_unk = sum((xs == UNK_ID).sum() for xs, ts in data)
        total = sum(xs.size for xs, ts in data)
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
    parser.add_argument('--attn', default='global', choices=['disuse', 'global'], help='Type of attention mechanism')
    parser.add_argument('--encoder', default='LSTM', choices=['LSTM', 'GRU', 'CNN'], help='Type of Decoder NN')
    parser.add_argument('--n_encoder', type=int, default=2, help='Number of Encoders')
    parser.add_argument('--score', default='dot', choices=['dot', 'general', 'concat'], help=' ')
    parser.add_argument('--kana', default=False, action='store_true', help='Whether to convert to kana')
    parser.add_argument('--emb', default=None, help='Pretrained word embedding file')
    parser.add_argument('--train', required=True, help='Train dataset file')
    parser.add_argument('--valid', required=True, help='Validation dataset file')
    parser.add_argument('--save_dir', required=True, help='Directory to save results')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    # prepare
    train, converters = make_dataset(args.train, vocab_size=args.vocabsize, min_freq=args.minfreq,
                                     n_encoder=args.n_encoder, to_kana=args.kana, emb=args.emb)
    w2id, class2id, initialW = converters['w2id'], converters['class2id'], converters['initialW']
    valid, _ = make_dataset(args.valid, w2id, class2id, n_encoder=args.n_encoder, to_kana=args.kana)
    n_vocab = len(w2id)
    print('n_vocab', n_vocab)
    print('initialW.shape', initialW.shape)
    n_class = len(class2id)
    unk_rate = unknown_rate(train)
    vocab = {'class2id': class2id, 'w2id': w2id}
    args.__dict__['train_size'] = len(train)
    args.__dict__['unknown_rate'] = unk_rate
    os.makedirs(args.save_dir, exist_ok=True)
    json.dump(vocab, open(args.save_dir + '/vocab.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(args.__dict__, open(args.save_dir + '/opts.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print('Train size:', len(train))
    print('Vocab size:', n_vocab)
    print('Unknown rate: {:.2f}%'.format(unk_rate * 100))

    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, batch_size=args.batchsize,
                                                  repeat=False, shuffle=False)

    # model
    if args.encoder == 'CNN' and args.n_encoder == 1:
        model = nets.Classifier(n_vocab, args.unit, n_class, args.layer, args.dropout, args.encoder, initialW)
    elif args.encoder == 'CNN':
        model = nets.ContextClassifier(n_vocab, args.unit, n_class, args.layer, args.dropout, args.encoder, initialW)
    elif args.attn == 'disuse':
        model = nets.ContextClassifier(n_vocab, args.unit, n_class, args.layer, args.dropout, args.encoder, initialW)
    elif args.attn == 'global':
        model = nets.AttnContextClassifier(n_vocab, args.unit, n_class, args.layer, args.dropout, args.encoder, args.score, initialW)
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
