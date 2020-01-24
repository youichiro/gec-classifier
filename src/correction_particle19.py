# -*- coding: utf-8 -*-
import json
import copy
import chainer
from nets import Classifier, ContextClassifier, AttnContextClassifier
from mecab import Mecab
from utils import make_dataset, clean_text, convert_to_kana, TARGETS, TARGET_PARTS, get_target_positions, get_complement_positions
from .train import seq_convert


class Checker:
    def __init__(self, mecab_dict_file, model_file, vocab_file, opts_file, lm_data=False, reverse=False, threshold=0.0):
        # mecab
        self.mecab = Mecab(mecab_dict_file)

        # LM
        if lm_data:
            from calculator import LM
            self.lm = LM(lm_data)

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
        self.threshold = threshold

    def _preprocess(self, text):
        """正規化，形態素解析，かな変換を行う"""
        text = clean_text(text)  # 全角→半角，数字→#
        org_words, parts = self.mecab.tagger(text)
        org_words, parts = self.mecab.preprocessing_to_particle(org_words, parts, TARGETS, TARGET_PARTS)

        assert org_words is not None

        if self.to_kana:
            words = convert_to_kana(' '.join(org_words)).split(' ')  # かな変換
            assert len(org_words) == len(words), f"\norg_words: {org_words}\nwords: {words}"
        else:
            words = org_words[::]
        return org_words, words, parts

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
                score = scores[predict]
        predict = self.id2class.get(int(predict))
        return predict, score

    def _predict_lm(self, test_data):
        """言語モデル確率の最も高い単語と文確率を返す"""
        if self.n_encoder == 2:
            pass
        else:
            xs = test_data[0][0]
            words = [self.id2w[i] for i in xs]
            target_id = words.index('<TARGET>')
            left_words = words[:target_id]
            right_words = words[target_id+1:]
            best = ['', -10000000000000]
            for t in TARGETS:
                t = '' if t == 'DEL' else t
                words = left_words + [t] + right_words
                lm_score = self.lm.probability(words)
                if lm_score > best[1]:
                    best = [t, lm_score]
            if best[0] == '':
                best[0] = 'DEL'
            return best


    def correction(self, text):
        """訂正文を返す"""
        org_words, words, parts = self._preprocess(text)
        target_idx = get_target_positions(words, parts)
        comp_idx = get_complement_positions(words, parts)
        # 重複がないことを保証
        assert set(target_idx) & set(comp_idx) == set()
        all_idx = sorted(target_idx + comp_idx)
        add_count = 0

        if self.reverse:
            all_idx = all_idx[::-1]  # 文末から訂正

        for idx in all_idx:
            idx += add_count  # 挿入した数だけ右にずらす

            # 置換 or 削除
            if idx in target_idx:
                left_text, right_text = ' '.join(words[:idx]), ' '.join(words[idx+1:])
                labeled_sentence = f'{left_text} <DEL> {right_text}'  # <DEL>に意味はない
                test_data, _ = make_dataset([labeled_sentence], self.w2id, self.class2id,
                                            n_encoder=self.n_encoder, to_kana=self.to_kana, is_train=False)
                if self.lm:
                    predict, score = self._predict_lm(test_data)
                else:
                    predict, score = self._predict(test_data)
                    predict = words[idx] if score < self.threshold else predict  # 予測確率が閾値より下なら変えない
                if predict == 'DEL':
                    # 左にシフト
                    words = words[:idx] + words[idx+1:]
                    org_words = org_words[:idx] + org_words[idx+1:]
                    add_count -= 1
                    target_idx = [idx-1 for idx in target_idx]
                else:
                    words[idx] = predict
                    org_words[idx] = predict
            # 挿入 or キープ
            else:
                left_text, right_text = ' '.join(words[:idx]), ' '.join(words[idx:])
                labeled_sentence = f'{left_text} <DEL> {right_text}'
                test_data, _ = make_dataset([labeled_sentence], self.w2id, self.class2id,
                                            n_encoder=self.n_encoder, to_kana=self.to_kana, is_train=False)
                if self.lm:
                    predict, score = self._predict_lm(test_data)  # 言語モデルで予測
                else:
                    predict, score = self._predict(test_data)
                    predict = 'DEL' if score < self.threshold else predict  # 予測確率が閾値より下なら変えない

                if predict == 'DEL':
                    pass  # キープ
                else:  # 挿入
                    # 右にシフト
                    words = words[:idx] + [predict] + words[idx:]
                    org_words = org_words[:idx] + [predict] + org_words[idx:]
                    add_count += 1
                    target_idx = [idx+1 for idx in target_idx]

        corrected_sentence = ''.join(org_words)
        corrected_sentence = corrected_sentence.replace('DEL', '')
        return corrected_sentence


    def correction_api(self, text):
        if not text:
            return ''
        org_words, words, parts = self._preprocess(text)
        input_words = copy.deepcopy(org_words)
        target_idx = get_target_positions(words, parts)
        comp_idx = get_complement_positions(words, parts)
        assert set(target_idx) & set(comp_idx) == set()
        all_idx = sorted(target_idx + comp_idx)
        add_count = 0
        replaces = []
        adds = []
        dels = []

        if self.reverse:
            all_idx = all_idx[::-1]  # 文末から訂正

        for idx in all_idx:
            idx += add_count  # 挿入した数だけ右にずらす

            # 置換 or 削除
            if idx in target_idx:
                left_text, right_text = ' '.join(
                    words[:idx]), ' '.join(words[idx+1:])
                # <DEL>に意味はない
                labeled_sentence = f'{left_text} <DEL> {right_text}'
                test_data, _ = make_dataset([labeled_sentence], self.w2id, self.class2id,
                                            n_encoder=self.n_encoder, to_kana=self.to_kana, is_train=False)
                if self.lm:
                    predict, score = self._predict_lm(test_data)
                else:
                    predict, score = self._predict(test_data)
                    predict = words[idx] if score < self.threshold else predict
                if predict == 'DEL':
                    # 左にシフト
                    words = words[:idx] + words[idx+1:]
                    org_words = org_words[:idx] + org_words[idx+1:]
                    dels.append(idx-add_count)
                    add_count -= 1
                    target_idx = [idx-1 for idx in target_idx]
                else:
                    if org_words[idx] != predict:
                        words[idx] = predict
                        org_words[idx] = predict
                        replaces.append(idx)
            # 挿入 or キープ
            else:
                left_text, right_text = ' '.join(
                    words[:idx]), ' '.join(words[idx:])
                labeled_sentence = f'{left_text} <DEL> {right_text}'
                test_data, _ = make_dataset([labeled_sentence], self.w2id, self.class2id,
                                            n_encoder=self.n_encoder, to_kana=self.to_kana, is_train=False)
                if self.lm:
                    predict, score = self._predict_lm(test_data)  # 言語モデルで予測
                else:
                    predict, score = self._predict(test_data)
                    predict = 'DEL' if score < self.threshold else predict  # 予測確率が閾値より下なら変えない

                if predict == 'DEL':
                    pass  # キープ
                else:  # 挿入
                    # 右にシフト
                    words = words[:idx] + [predict] + words[idx:]
                    org_words = org_words[:idx] + [predict] + org_words[idx:]
                    add_count += 1
                    target_idx = [idx+1 for idx in target_idx]
                    adds.append(idx)

        return {
            'input_words': input_words,
            'corrected_words': org_words,
            'replaces': replaces,
            'adds': adds,
            'dels': dels
        }
