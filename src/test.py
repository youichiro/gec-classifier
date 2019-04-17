# -*- coding: utf-8 -*-
import json
import argparse
import nets
from train import seq_convert
from utils import make_dataset, tagging
import chainer
from chainer.dataset import convert
from tqdm import tqdm


def test_on_pair_encoder(model, test, id2w, id2class, do_show):
    if do_show:
        print('text\tanswer\tpredict\tresult')
    count, t = 0, 0
    for i in tqdm(range(len(test))):
        lxs, rxs, ts = seq_convert([test[i]])
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            predict = model.predict(lxs, rxs, argmax=True)
        left_text = ' '.join([id2w.get(int(idx), '<UNK>') for idx in lxs[0]])
        right_text = ' '.join([id2w.get(int(idx), '<UNK>') for idx in rxs[0]])
        target = id2class.get(int(ts[0]))
        predict = id2class.get(int(predict[0]))
        result = 1 if predict == target else 0
        count += 1
        t += result
        if do_show:
            print(f'{left_text} <TARGET> {right_text}\t{target}\t{predict}\t{result}')
    print('\nAccuracy {:.2f}% ({}/{})'.format(t / count * 100, t, count))


def test_on_single_encoder(model, test, id2w, id2class, do_show):
    if do_show:
        print('text\tanswer\tpredict\tresult')
    total_predict, accurate = 0, 0
    total_particles = {'が': 0, 'を': 0, 'に': 0, 'で': 0}
    accurate_particles = {'が': 0, 'を': 0, 'に': 0, 'で': 0}

    for i in tqdm(range(len(test))):
        xs, ts = seq_convert([test[i]])
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            predict = model.predict(xs, argmax=True)
        text = ' '.join([id2w.get(int(idx), '<UNK>') for idx in xs[0]])
        answer = id2class.get(int(ts[0]))
        predict = id2class.get(int(predict[0]))
        total_predict += 1
        accurate += 1 if predict == answer else 0

        # 格助詞別スコア
        total_particles[answer] += 1
        if is_correct:
            accurate_particles[answer] += 1

        if do_show:
            print(f'{text}\t{answer}\t{predict}\t{predict == answer}')

    print('\nAccuracy {:.2f}% ({}/{})'.format(accurate / total_predict * 100, accurate, total_predict))


def load_model():
    # args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', required=True, help='Directory of trained models')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch of model to use')
    parser.add_argument('--err', required=True, help='Segmented error text file')
    parser.add_argument('--ans', required=True, help='Segmented answer text file')
    parser.add_argument('--show', default=False, action='store_true', help='Whether to show results')
    args = parser.parse_args()

    # prepare
    vocab = json.load(open(args.model_dir + '/vocab.json'))
    w2id = vocab['w2id']
    class2id = vocab['class2id']
    id2w = {v: k for k, v in w2id.items()}
    id2class = {v: k for k, v in class2id.items()}
    n_vocab = len(w2id)
    n_class = len(class2id)
    opts = json.load(open(args.model_dir + '/opts.json'))
    n_emb = opts['n_emb']
    n_units = opts['unit']
    n_layer = opts['layer']
    dropout = opts['dropout']
    score = opts.get('score', 'dot')
    encoder = opts['encoder']
    n_encoder = int(opts.get('n_encoder', 2))
    attn = opts['attn']
    to_kana = opts.get('kana', False)
    model_file = args.model_dir + '/model-e{}.npz'.format(args.epoch)

    # model
    if encoder == 'CNN' and n_encoder == 1:
        model = nets.Classifier(n_vocab, n_emb, n_units, n_class, n_layer, dropout, encoder)
    elif encoder == 'CNN':
        model = nets.ContextClassifier(n_vocab, n_emb, n_units, n_class, n_layer, dropout, encoder)
    elif attn == 'disuse':
        model = nets.ContextClassifier(n_vocab, n_emb, n_units, n_class, n_layer, dropout, encoder)
    elif attn == 'global':
        model = nets.AttnContextClassifier(n_vocab, n_emb, n_units, n_class, n_layer, dropout, encoder, score)
    chainer.serializers.load_npz(model_file, model)

    # test
    err_data = open(args.err).readlines()
    ans_data = open(args.ans).readlines()
    testdata = [tagging(err, ans) for err, ans in zip(err_data, ans_data)
                if len(err) == len(ans) and err != ans]
    test_data, _ = make_dataset(testdata, w2id, class2id, n_encoder=n_encoder, to_kana=to_kana)

    return model, test_data, id2w, id2class, n_encoder, args.show


def main():
    model, test_data, id2w, id2class, n_encoder, do_show = load_model()
    if n_encoder == 2:
        test_on_pair_encoder(model, test_data, id2w, id2class, do_show)
    elif n_encoder == 1:
        test_on_single_encoder(model, test_data, id2w, id2class, do_show)


if __name__ == '__main__':
    main()
