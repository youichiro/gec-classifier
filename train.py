import os
import json
import numpy
import argparse
import nets
from utils import make_dataset, IGNORE_ID
import chainer
from chainer.dataset import convert
from chainer.backends import cuda
from chainer.training import extensions


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
    lxs, rxs, ts = zip(*batch)
    lxs_block = convert.concat_examples(lxs, device, padding=IGNORE_ID)
    rxs_block = convert.concat_examples(rxs, device, padding=IGNORE_ID)
    ts_block = convert.concat_examples(ts, device)
    return (lxs_block, rxs_block, ts_block)


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=300)
    parser.add_argument('--epoch', '-e', type=int, default=30)
    parser.add_argument('--gpuid', '-g', type=int, default=-1)
    parser.add_argument('--save_dir', '-s', required=True, help='Directory to save results')
    parser.add_argument('--unit', '-u', type=int, default=300)
    parser.add_argument('--layer', '-l', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.1)
    parser.add_argument('--train', '-t', required=True, help='Train dataset file')
    parser.add_argument('--valid', '-v', required=True, help='Validation dataset file')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    # prepare
    train, w2id, classes = make_dataset(args.train)
    valid, _, _ = make_dataset(args.valid, w2id, classes)
    n_vocab = len(w2id)
    n_class = len(classes)
    vocab = {'classes': classes, 'w2id': w2id}
    os.makedirs(args.save_dir, exist_ok=True)
    json.dump(vocab, open(args.save_dir + '/vocab.json', 'w'), ensure_ascii=False)
    json.dump(args.__dict__, open(args.save_dir + '/opts.json', 'w'))

    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, batch_size=args.batchsize,
                                                  repeat=False, shuffle=False)

    # model
    left_encoder = nets.RNNEncoder(n_vocab, args.unit, args.layer, args.dropout)
    right_encoder = nets.RNNEncoder(n_vocab, args.unit, args.layer, args.dropout)
    model = nets.ContextClassifier(left_encoder, right_encoder, n_class)
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
