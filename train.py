import os
import numpy
import pickle
import nets
from utils import make_dataset, IGNORE_ID
import chainer
from chainer.dataset import convert
from chainer.backends import cuda
from chainer.training import extensions

n_layers = 1
n_units = 300
dropout = 0.1
max_epoch = 30
train_path = 'datasets/train.txt'
valid_path = 'datasets/dev.txt'
save_dir = 'mai_all'
gpuid = -1
log_interval = 1


class SaveModel(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir

    def __call__(self, trainer):
        model_name = 'model-e{}.npz'.format(trainer.updater.epoch)
        save_path = os.path.join(self.save_dir, model_name)
        chainer.serializers.save_npz(save_path, self.model)


def seq_convert(batch, device=None):
    lxs, rxs, ts = zip(*batch)
    lxs_block = convert.concat_examples(lxs, device, padding=IGNORE_ID)
    rxs_block = convert.concat_examples(rxs, device, padding=IGNORE_ID)
    return (lxs_block, rxs_block, ts)


def main():
    train, w2id, classes = make_dataset(train_path)
    valid, _, _ = make_dataset(valid_path, w2id, classes)
    n_vocab = len(w2id)
    n_class = len(classes)
    holds = {'vocab': w2id, 'classes': classes}
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(holds, open(save_dir + '/holds.pkl', 'wb'))

    train_iter = chainer.iterators.SerialIterator(train, batch_size=3)
    valid_iter = chainer.iterators.SerialIterator(valid, batch_size=3, repeat=False, shuffle=False)

    left_encoder = nets.RNNEncoder(n_vocab, n_units, n_layers, dropout)
    right_encoder = nets.RNNEncoder(n_vocab, n_units, n_layers, dropout)
    model = nets.ContextClassifier(left_encoder, right_encoder, n_class)
    if gpuid >= 0:
        cuda.get_device_from_id(gpuid).use()
        model.to_gpu(gpuid)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = chainer.training.StandardUpdater(train_iter, optimizer,
                                               converter=seq_convert, device=gpuid)

    trainer = chainer.training.Trainer(updater, (max_epoch, 'epoch'), out=save_dir)
    trainer.extend(extensions.Evaluator(valid_iter, model, converter=seq_convert, device=gpuid))
    trainer.extend(SaveModel(model, save_dir))
    trainer.extend(extensions.LogReport(trigger=(log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/accuracy',
        'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']), trigger=(log_interval, 'iteration'))
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
