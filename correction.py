import json
import argparse
import chainer
from nets import CNNEncoder, Classifier, ContextClassifier, AttnContextClassifier
from mecab import Mecab
from utils import make_dataset, normalize_text, convert_to_kana
from train import seq_convert



class Model:
    def __init__(self, mecab_dict_file, model_file, vocab_file, opts_file):
        # mecab
        self.target_lemma = ['が', 'を', 'に', 'で']
        self.target_pos = '助詞-格助詞'
        self.mecab = Mecab(mecab_dict_file)

        # prepare
        vocab = json.load(open(vocab_file))
        w2id, class2id = vocab['w2id'], vocab['class2id']
        id2w = {v: k for k, v in w2id.items()}
        id2class = {v: k for k, v in class2id.items()}
        n_vocab = len(w2id)
        n_class = len(class2id)
        opts = json.load(open(opts_file))
        n_emb = opts['n_emb']
        n_unit = opts['unit']
        n_layer = opts['layer']
        dropout = opts['dropout']
        score = opts['score']
        encoder = opts['encoder']
        n_encoder = int(opts['n_encoder'])
        attn = opts['attn']
        to_kana = opts['kana']

        # model
        if encoder == 'CNN' and n_encoder == 1:
            model = Classifier(n_vocab, n_emb, n_unit, n_class, n_layer, dropout, encoder)
        elif encoder == 'CNN':
            model = ContextClassifier(n_vocab, n_emb, n_unit, n_class, n_layer, dropout, encoder)
        elif attn == 'disuse':
            model = nets.ContextClassifier(n_vocab, n_emb, n_unit, n_class, n_layer, dropout, encoder)
        elif attn == 'global':
            model = nets.AttnContextClassifier(n_vocab, n_emb, n_unit, n_class, n_layer, dropout, encoder, score)
        chainer.serializers.load_npz(model_file, model)

        self.model = model
        self.w2id = w2id
        self.class2id = class2id
        self.id2w = id2w
        self.id2class = id2class
        self.n_encoder = n_encoder
        self.to_kana = to_kana

    def get_target_positions(self, words, parts):
        target_idx = [i for i, (w, p) in enumerate(zip(words, parts))
                      if p == self.target_pos and w in self.target_lemma \
                      and i != 0 and i != len(words) - 1]  # 文頭と文末の格助詞は除く
        return target_idx

    def predict(self, test_data):
        if self.n_encoder == 2:
            lxs, rxs, ts = seq_convert(test_data)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                predict = self.model.predict(lxs, rxs, argmax=True)
            predict = self.id2class.get(int(predict[0]))
        else:
            xs, ts = seq_convert(test_data)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                predict = self.model.predict(xs, argmax=True)
            predict = self.id2class.get(int(predict[0]))
        return predict

    def correction(self, sentence):
        sentence = normalize_text(sentence)  # 全角→半角，数字→#
        org_words, parts = self.mecab.tagger(sentence)  # 形態素解析
        if self.to_kana:
            words = convert_to_kana(' '.join(org_words)).split(' ')  # カナ変換
            if len(org_words) != len(words):
                return "error"
        else:
            words = org_words[::]

        target_idx = self.get_target_positions(words, parts)  # 訂正対象の位置リスト
        for idx in target_idx:
            marked_sentence = '{} <{}> {}'.format(
                ' '.join(words[:idx]), words[idx], ' '.join(words[idx+1:]))  # 格助詞を<>で囲む
            test_data, _ = make_dataset([marked_sentence], self.w2id, self.class2id,
                                        n_encoder=self.n_encoder, to_kana=self.to_kana)
            predict = self.predict(test_data)
            words[idx] = predict  # 予測に置換
            org_words[idx] = predict
        corrected = ''.join(org_words)
        return corrected
