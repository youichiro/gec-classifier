# -*- coding: utf-8 -*-
import json
import chainer
from nets import Classifier, ContextClassifier, AttnContextClassifier
from mecab import Mecab
from utils import make_dataset, clean_text, convert_to_kana
from train import seq_convert


TARGETS = ['が', 'の', 'を', 'に', 'へ', 'と', 'より', 'から', 'で', 'や',
           'は', 'には', 'からは', 'とは', 'では', 'へは', 'までは', 'よりは', 'まで', 'DEL']  # 19種類+削除
TARGET_PARTS = ['助詞-格助詞', '助詞-副助詞', '助詞-係助詞', '助詞-接続助詞',
                '助詞-終助詞', '助詞-準体助詞', '助詞']  # '助詞'はオリジナル設定


class Checker:
    def __init__(self, mecab_dict_file, model_file, vocab_file, opts_file, reverse=False):
        # mecab
        self.mecab = Mecab(mecab_dict_file)

        # prepare model
        vocab = json.load(open(vocab_file, encoding='utf-8'))
        w2id, class2id = vocab['w2id'], vocab['class2id']
        id2w = {v: k for k, v in w2id.items()}
        id2class = {v: k for k, v in class2id.items()}
        n_vocab = len(w2id)
        n_class = len(class2id)
        opts = json.load(open(opts_file, encoding='utf-8'))
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
            model = ContextClassifier(n_vocab, n_emb, n_unit, n_class, n_layer, dropout, encoder)
        elif attn == 'global':
            model = AttnContextClassifier(n_vocab, n_emb, n_unit, n_class, n_layer, dropout, encoder, score)
        chainer.serializers.load_npz(model_file, model)

        self.model = model
        self.w2id = w2id
        self.class2id = class2id
        self.id2w = id2w
        self.id2class = id2class
        self.n_encoder = n_encoder
        self.to_kana = to_kana
        self.reverse = reverse

    def _preprocess(self, text):
        """正規化，形態素解析，かな変換を行う"""
        text = clean_text(text)  # 全角→半角，数字→#
        org_words, parts = self.mecab.tagger(text)
        org_words, parts = self.mecab.preprocessing_to_particle(org_words, parts, TARGETS, TARGET_PARTS)

        assert org_words is not None

        if self.to_kana:
            words = convert_to_kana(' '.join(org_words)).split(' ')  # かな変換
            assert len(org_words) == len(words)
        else:
            words = org_words[::]
        return org_words, words, parts

    def _get_target_positions(self, words,  parts):
        """訂正対象の位置を返す"""
        target_idx = [i for i, (w, p) in enumerate(zip(words, parts))
                      if p in TARGET_PARTS and w in TARGETS \
                      and i != 0 and i != len(words) - 1]  # 文頭と文末は除く
        return target_idx

    def _predict(self, test_data):
        """予測単語と確率を返す"""
        if self.n_encoder == 2:
            lxs, rxs, _ = seq_convert(test_data)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                predict = self.model.predict(lxs, rxs, argmax=True)[0]
                scores = self.model.predict(lxs, rxs, softmax=True)[0]
        else:
            xs, _ = seq_convert(test_data)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                predict = self.model.predict(xs, argmax=True)[0]
                scores = self.model.predict(xs, softmax=True)[0]
        predict = self.id2class.get(int(predict))
        return predict, scores

    def correction(self, text):
        """訂正文を返す"""
        org_words, words, parts = self._preprocess(text)
        target_idx = self._get_target_positions(words, parts)

        if self.reverse:
            target_idx = target_idx[::-1]  # 文末から訂正

        for idx in target_idx:
            marked_sentence = '{} <{}> {}'.format(
                ' '.join(words[:idx]), words[idx], ' '.join(words[idx+1:])  # 訂正対象を<>で囲む
            )
            test_data, _ = make_dataset([marked_sentence], self.w2id, self.class2id,
                                        n_encoder=self.n_encoder, to_kana=self.to_kana, is_train=False)
            predict, _ = self._predict(test_data)
            words[idx] = predict
            org_words[idx] = predict

        corrected_sentence = ''.join(org_words)
        corrected_sentence = corrected_sentence.replace('DEL', '')
        return corrected_sentence
