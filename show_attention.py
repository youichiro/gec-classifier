import chainer
from train import seq_convert
from test import load_model


def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)


def make_html(words, scores):
    html = ""
    for word, score in zip(words, scores):
        html += ' ' + highlight(word, score)
    return html + "<br><br>\n"


def show(model, test, id2w, id2class):
    filename = 'show_naist.html'
    with open(filename, 'w') as f:
        f.write(' ')
    # for i in range(len(test)):
    for i in range(1):
        lxs, rxs, ts = seq_convert([test[i]])
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            predict = model.predict(lxs, rxs, argmax=True)
        left_words = [id2w.get(int(idx), '') for idx in lxs[0]]
        right_words = [id2w.get(int(idx), '') for idx in rxs[0]]
        target = id2class.get(int(ts[0]))
        predict = id2class.get(int(predict[0]))
        # print('{} [{} {}] {}'.format(' '.join(left_words), predict, target, ' '.join(right_words)))
        left_scores = model.left_attn.scores[0].data.tolist()
        right_scores = model.right_attn.scores[0].data.tolist()[::-1]
        words = left_words + ['[{} {}]'.format(predict, target)] + right_words
        scores = left_scores + [0] + right_scores
        result = predict == target
        html = make_html(words, scores)
        with open(filename, 'a') as f:
            f.write('<b>{}</b>\n'.format(result))
            f.write(html + '\n')


if __name__ == '__main__':
    show(*load_model())
