import MeCab


class Mecab:
    def __init__(self, dict_file):
        self.t = MeCab.Tagger('-d {}'.format(dict_file))

    def tagger(self, text):
        n = self.t.parse(text)
        lines = n.split('\n')
        words, parts = [], []
        for line in lines[:-2]:
            words.append(line.split('\t')[0])
            parts.append(line.split('\t')[4])
        return words, parts
