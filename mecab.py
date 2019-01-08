import MeCab


class Mecab:
    def __init__(self, dict_path):
        self.t = MeCab.Tagger('-d {}'.format(dict_path))

    def tagger(self, text):
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
