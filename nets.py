import numpy
import chainer
import chainer.functions as F
import chainer.links as L
from utils import IGNORE_ID


def sequence_embed(embed, xs, dropout=0.1):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Encoder(chainer.Chain):
    def __init__(self, n_vocab, n_units, n_layers=1, dropout=0.1):
        super().__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=None, ignore_label=IGNORE_ID)
            self.rnn = L.NStepLSTM(n_layers, n_units, n_units, dropout)
        self.n_layers = n_layers
        self.out_units = n_units
        self.dropout = dropout

    def __call__(self, xs):
        exs = sequence_embed(self.embed, xs, self.dropout)
        last_h, last_c, ys = self.rnn(None, None, exs)
        assert (last_h.shape == (self.n_layers, len(xs), self.out_units))
        concat_outputs = last_h[-1]
        return concat_outputs


class AttnEncoder(Encoder):
    def __call__(self, xs):
        exs = sequence_embed(self.embed, xs, self.dropout)
        hx, cx, oxs = self.rnn(None, None, exs)
        return hx, cx, oxs


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
    def __init__(self, left_encoder, right_encoder, n_units, n_class, dropout=0.1):
        super().__init__()
        with self.init_scope():
            self.left_encoder = left_encoder
            self.right_encoder = right_encoder
            self.output = L.Linear(n_units + n_units, n_class)
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
        rxs = rxs[:, ::-1]
        left_encodings = F.dropout(self.left_encoder(lxs), ratio=self.dropout)
        right_encodings = F.dropout(self.right_encoder(rxs), ratio=self.dropout)
        concat_encodings = F.concat((left_encodings, right_encodings))
        concat_outputs = self.output(concat_encodings)
        if softmax:
            return F.softmax(concat_outputs).data
        elif argmax:
            return self.xp.argmax(concat_outputs.data, axis=1)
        else:
            return concat_outputs


class GlobalAttention(chainer.Chain):
    def __init__(self, n_units, score):
        super().__init__()
        with self.init_scope():
            self.n_units = n_units
            self.score = score

            if score == 'general':
                self.wg = L.Linear(n_units, n_units)
            elif score == 'concat':
                self.wa = L.Linear(2*n_units, 1)

    def __call__(self, oxs, oys):
        self.bs, self.xlen, _ = oxs.shape
        _, self.ylen, _ = oys.shape

        if self.score == 'dot':
            scores = self.dot(oys, oxs)
        elif self.score == 'general':
            scores = self.general(oys, oxs)

        scores = F.broadcast_to(scores, (self.n_units, self.bs, self.xlen))
        scores = F.transpose(scores, (1, 2, 0))  # scores: (bs, xlen, unit)
        ct = F.sum(oxs * scores, axis=1)  # ct: (bs, unit)
        return ct

    def dot(self, oxs, oys):
        scores = F.sum(oxs * oys, axis=2)  # scores: (bs, xlen)
        scores = F.softmax(scores, axis=1)  # scores: (bs, xlen)
        return scores

    def general(self, oxs, oys):
        oys = F.stack(sequence_embed(self.wg, oys))
        scores = self.dot(oxs, oys)
        return scores


class AttnContextClassifier(chainer.Chain):
    def __init__(self, n_vocab, n_units, n_class, n_layers=1, dropout=0.1):
        super().__init__()
        with self.init_scope():
            self.left_encoder = AttnEncoder(n_vocab, n_units, n_layers, dropout)
            self.right_encoder = AttnEncoder(n_vocab, n_units, n_layers, dropout)
            self.left_attn = GlobalAttention(n_units, score='dot')
            self.right_attn = GlobalAttention(n_units, score='dot')
            self.wc = L.Linear(2*n_units, n_units)
            self.wo = L.Linear(n_units, n_class)
        self.n_units = n_units
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
        rxs = rxs[:, ::-1]
        _, _, los = self.left_encoder(lxs)
        _, _, ros = self.right_encoder(rxs)
        los = F.stack(los)
        ros = F.stack(ros)
        lstate = self.left_attn(los, self.make_oys(los))  # lstate: (bs, n_units)
        rstate = self.right_attn(ros, self.make_oys(ros))  # rstate: (bs, n_units)

        state = F.concat((lstate, rstate), axis=1)  # state: (bs, 2*n_units)
        relu_state = F.relu(F.stack(self.wc(state)))  # relu_state: (bs, n_units)
        concat_outputs = F.stack(self.wo(relu_state))  # concat_outputs: (bs, n_class)
        if softmax:
            return F.softmax(concat_outputs).data
        elif argmax:
            return self.xp.argmax(concat_outputs.data, axis=1)
        else:
            return concat_outputs

    def make_oys(self, oxs):
        bs, xlen, _ = oxs.shape  # oxs: (bs, xlen, n_units)
        oxs_last = oxs[::, -1]  # 最後の列の値の配列 oxs_last: (bs, u_units)
        oys = F.broadcast_to(oxs_last, (xlen, bs, self.n_units))  # xlenだけ伸ばす oys: (xlen, bs, n_units)
        oys = F.transpose(oys, (1, 0, 2))  # 転置 oys: (bs, xlen, n_units)
        return oys
