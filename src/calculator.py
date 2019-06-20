import kenlm

class LM:
    def __init__(self, model_file):
        print('Loading KenLM model:', model_file)
        self.lm = kenlm.Model(model_file)

    def probability(self, words):
        words = [word for word in words if word]
        return self.lm.score(' '.join(words), bos=True, eos=True)
