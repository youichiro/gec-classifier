# -*- coding: utf-8 -*-
import json
import argparse
import chainer
from nets import CNNEncoder, Classifier, ContextClassifier, AttnContextClassifier
from mecab import Mecab
from utils import make_dataset, normalize_text, convert_to_kana
from train import seq_convert


class Checker:
    def __init__(self, mecab_dict_file, model_file, vocab_file, opts_file, reverse=True, show=False):
        # mecab
        self.target_lemma = ['が', 'を', 'に', 'で']
        self.target_pos = ['助詞-格助詞', '助動詞']
        self.mecab = Mecab(mecab_dict_file)

        # prepare
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

        # for test
        self.show = show
        self.accurate = 0
        self.accurate_of_error = 0
        self.total_predict_num = 0
        self.total_error_num = 0
        self.num_sentence = 0
        self.error = 0
        self.target_statistic = []
        self.confusion_target_to_answer = {}
        self.confusion_target_to_predict = {}
        self.confusion_predict_to_answer = {}

    def _preprocess(self, sentence):
        """正規化，形態素解析，カナ変換を行う"""
        sentence = normalize_text(sentence)  # 全角→半角，数字→#
        org_words, parts = self.mecab.tagger(sentence)  # 形態素解析
        if self.to_kana:
            words = convert_to_kana(' '.join(org_words)).split(' ')  # カナ変換
            if len(org_words) != len(words):
                return None, None, None
        else:
            words = org_words[::]
        return org_words, words, parts

    def _get_target_positions(self, words, parts):
        """訂正対象の位置を返す"""
        target_idx = [i for i, (w, p) in enumerate(zip(words, parts))
                      if p in self.target_pos and w in self.target_lemma \
                      and i != 0 and i != len(words) - 1]  # 文頭と文末の格助詞は除く
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

    def correction(self, sentence):
        """訂正文を返す"""
        org_words, words, parts = self._preprocess(sentence)
        if org_words is None:
            return "error"

        target_idx = self._get_target_positions(words, parts)  # 訂正対象の位置リスト
        target_idx = target_idx[::-1]  # 後ろから訂正
        for idx in target_idx:
            marked_sentence = '{} <{}> {}'.format(
                ' '.join(words[:idx]), words[idx], ' '.join(words[idx+1:]))  # 格助詞を<>で囲む
            test_data, _ = make_dataset([marked_sentence], self.w2id, self.class2id,
                                        n_encoder=self.n_encoder, to_kana=self.to_kana)
            predict, _ = self._predict(test_data)
            words[idx] = predict  # 予測に置換
            org_words[idx] = predict
        corrected = ''.join(org_words)
        return corrected

    def correction_test(self, err, ans):
        """訂正して正解率を求める"""
        # 前処理
        err_org_words, err_words, err_parts = self._preprocess(err)
        ans_org_words, ans_words, _ = self._preprocess(ans)
        if err_org_words is None or ans_org_words is None:
            return "error"
        err_word_lens = [len(t) for t in err_words]
        ans_word_lens = [len(t) for t in ans_words]

        # もし誤り文と正解文で単語分割の対応が異なったらエラーとする
        if err_word_lens != ans_word_lens:
            # print('error\n' + ' '.join(err_words) + '\n' + ' '.join(ans_words) + '\n')
            self.error += 1
            return ''.join(err_org_words)

        self.num_sentence += 1
        target_idx = self._get_target_positions(err_words, err_parts)  # 訂正対象の位置を取得
        error_idx = [idx for idx in range(len(err_words) - 1) if err_words[idx] != ans_words[idx]]  # 誤り箇所の位置を取得
        self.total_error_num += len(error_idx)  # 間違い箇所数
        self.total_predict_num += len(target_idx)  # 予測する箇所数

        # 誤り箇所が訂正対象に全て含まれていることを保証
        assert len(set(error_idx) - set(target_idx)) == 0

        # 後ろから訂正
        if self.reverse is True:
            target_idx = target_idx[::-1]

        predict_list = []
        for idx in target_idx:
            target = err_words[idx]  # target: 対象単語
            marked_sentence = '{} <{}> {}'.format(
                ' '.join(err_words[:idx]), err_words[idx], ' '.join(err_words[idx+1:]))  # 格助詞を<>で囲む
            test_data, _ = make_dataset([marked_sentence], self.w2id, self.class2id,
                                        n_encoder=self.n_encoder, to_kana=self.to_kana)
            predict, _ = self._predict(test_data)  # predict: 予測単語
            predict_list.append(predict)
            # 予測に置換
            err_words[idx] = predict
            err_org_words[idx] = predict
            answer = ans_words[idx]  # answer: 正解単語

            if predict == answer:
                self.accurate += 1  # 予測して，かつ正解
                if idx in error_idx:
                    self.accurate_of_error += 1  # 間違い箇所であり，かつ正解

            # 誤り箇所である条件の上で
            if idx in error_idx:
                # confusion matrix (target -> answer) を作成
                cell = f'{target}->{answer}'
                if cell in self.confusion_target_to_answer.keys():
                    self.confusion_target_to_answer[cell] += 1
                else:
                    self.confusion_target_to_answer[cell] = 1

                # confusion matrix (target -> predict) を作成
                cell = f'{target}->{predict}'
                if cell in self.confusion_target_to_predict.keys():
                    self.confusion_target_to_predict[cell] += 1
                else:
                    self.confusion_target_to_predict[cell] = 1

                # confusion matrix (predict -> answer) を作成
                cell = f'{predict}->{answer}'
                if cell in self.confusion_predict_to_answer.keys():
                    self.confusion_predict_to_answer[cell] += 1
                else:
                    self.confusion_predict_to_answer[cell] = 1

        # [1文中の対象格助詞の数，格助詞誤りの数]
        self.target_statistic.append([len(target_idx), len(error_idx)])

        corrected = ''.join(err_org_words)
        if self.show:
            print(f'{self.num_sentence}')
            print(f'err: {err}')
            print(f'ans: {ans}')
            print(f'out: {corrected}')
            print(f'Result: {ans == corrected}\n')
        return corrected


    def correction_for_api(self, sentence):
        """チェッカー用の訂正結果を返す"""
        sentence = normalize_text(sentence)  # 全角→半角，数字→#
        org_words, parts = self.mecab.tagger(sentence)  # 形態素解析
        if self.to_kana:
            words = convert_to_kana(' '.join(org_words)).split(' ')  # カナ変換
            if len(org_words) != len(words):
                return "error"
        else:
            words = org_words[::]

        fix_flags = [0] * len(words)
        score_list = [{}] * len(words)

        target_idx = self._get_target_positions(words, parts)  # 訂正対象の位置リスト
        target_idx = target_idx[::-1]  # 後ろから訂正
        for idx in target_idx:
            marked_sentence = '{} <{}> {}'.format(
                ' '.join(words[:idx]), words[idx], ' '.join(words[idx+1:]))  # 格助詞を<>で囲む
            test_data, _ = make_dataset([marked_sentence], self.w2id, self.class2id,
                                        n_encoder=self.n_encoder, to_kana=self.to_kana)
            predict, scores = self._predict(test_data)

            if words[idx] != predict:
                words[idx] = predict  # 予測に置換
                org_words[idx] = predict
                fix_flags[idx] = 1  # 置換フラグ
            else:
                fix_flags[idx] = 2  # 無修正フラグ

            score_dic = dict(zip(self.class2id.keys(), scores))
            if ' ' in score_dic.keys():
                del score_dic[' ']  # 謎の空白クラスを削除
            sorted_score_dic = dict(sorted(score_dic.items(), key=lambda x: x[1], reverse=True))
            d = {
                'keys': list(sorted_score_dic.keys()),
                'scores': [f'{score*100:.1f}' for score in sorted_score_dic.values()]
            }
            score_list[idx] = d

        return [[word, fix_flag, score] for word, fix_flag, score in zip(org_words, fix_flags, score_list)]


def test():
    mecab_dict_file = '/usr/local/lib/mecab/dic/unidic'
    model_file = '/Users/you_pro/workspace/jnlp/gec-classifier/src/models/bccwj+lang8_cnn1_single_unit500_nwjc_kana/model-e5.npz'
    vocab_file = '/Users/you_pro/workspace/jnlp/gec-classifier/src/models/bccwj+lang8_cnn1_single_unit500_nwjc_kana/vocab.json'
    opts_file = '/Users/you_pro/workspace/jnlp/gec-classifier/src/models/bccwj+lang8_cnn1_single_unit500_nwjc_kana/opts.json'
    checker = Checker(mecab_dict_file, model_file, vocab_file, opts_file)
    s = '彼に車に買う'
    crr = checker.correction_for_api(s)
    print(crr)
