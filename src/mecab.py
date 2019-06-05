# -*- coding: utf-8 -*-
import MeCab


class Mecab:
    def __init__(self, dict_path):
        self.t = MeCab.Tagger('-d {}'.format(dict_path))

    def tagger(self, text):
        """形態素解析して単語列と品詞列を返す"""
        n = self.t.parse(text)
        lines = n.split('\n')
        words, parts = [], []
        for line in lines[:-2]:
            words.append(line.split('\t')[0])
            parts.append(line.split('\t')[4])
        return words, parts

    def to_kana(self, text):
        n = self.t.parse(text)
        lines = n.split('\n')
        kanas = []
        for line in lines[:-2]:
            kanas.append(line.split('\t')[1])
        return kanas

    @staticmethod
    def preprocessing_to_particle(words, parts, targets, target_parts):
        """'よりは'のような助詞2単語を1単語にまとめる処理"""
        new_words, new_parts = [], []
        continue_flag = False
        for i, (w, p) in enumerate(zip(words, parts)):
            if continue_flag:
                continue_flag = False
                continue
            if p in target_parts and words[i] + words[i+1] in targets:
                new_words.append(words[i] + words[i+1])
                new_parts.append('助詞')
                continue_flag = True
            else:
                new_words.append(w)
                new_parts.append(p)
        return new_words, new_parts
