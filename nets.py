import numpy
import chainer
import chainer.functions as F
import chainer.links as L

embed_init = chainer.initializers.Uniform(.25)


def sequence_embed(embed, xs, dropout=0.):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


class RNNEncoder(chainer.Chain):
    def __init__(self, n_vocab, n_units, n_layers=1, dropout=0.1):
        super().__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=embed_init)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout)
        self.n_layers = n_layers
        self.out_units = n_units
        self.dropout = dropout

    def __call__(self, xs):
        exs = sequence_embed(self.embed, xs, self.dropout)
        last_h, last_c, ys = self.encoder(None, None, exs)
        assert (last_h.shape == (self.n_layers, len(xs), self.out_units))
        concat_outputs = last_h[-1]
        return concat_outputs


class Classifier(chainer.Chain):
    def __init__(self, encoder, n_class, dropout=0.1):
        super().__init__()
        with self.init_scope():
            self.encoder = encoder
            self.output = L.Linear(encoder.out_units, n_class)
        self.dropout = dropout

    def __call__(self, xs, ts):
        concat_outputs = self.predict(xs)
        concat_truths = F.concat(ts, axis=0)
        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        chainer.reporter.report({'loss': loss.data}, self)
        chainer.reporter.report({'accuracy': accuracy.data}, self)
        return loss

    def predict(self, xs, softmax=False, argmax=False):
        concat_encodings = F.dropout(self.encoder(xs), ratio=self.dropout)
        concat_outputs = self.output(concat_encodings)
        if softmax:
            return F.softmax(concat_outputs).data
        elif argmax:
            return self.xp.argmax(concat_outputs.data, axis=1)
        else:
            return concat_outputs


class ContextClassifier(chainer.Chain):
    def __init__(self, left_encoder, right_encoder, n_class, dropout=0.1):
        super().__init__()
        with self.init_scope():
            self.left_encoder = left_encoder
            self.right_encoder = right_encoder
            self.output = L.Linear(left_encoder.out_units + right_encoder.out_units, n_class)
        self.dropout = dropout

    def __call__(self, lxs, rxs, ts):
        concat_outputs = self.predict(lxs, rxs)
        concat_truths = F.concat(ts, axis=0)
        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        chainer.reporter.report({'loss': loss.data}, self)
        chainer.reporter.report({'accuracy': accuracy.data}, self)
        return loss

    def predict(self, lxs, rxs, softmax=False, argmax=False):
        left_encodings = F.dropout(self.left_encoder(lxs), ratio=self.dropout)
        right_encodings = F.dropout(self.right_encoder(rxs[:, ::-1]), ratio=self.dropout)
        concat_encodings = F.concat((left_encodings, right_encodings))
        concat_outputs = self.output(concat_encodings)
        if softmax:
            return F.softmax(concat_outputs).data
        elif argmax:
            return self.xp.argmax(concat_outputs.data, axis=1)
        else:
            return concat_outputs
