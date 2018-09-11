import pickle
import nets
from utils import make_dataset
from train import seq_convert
import chainer
from chainer.backends import cuda

save_dir = 'mai_all'
model_name = 'model-e10.npz'
test_path = 'datasets/dev.txt'
batch_size = 2
n_units = 300
n_layer = 1
dropout = 0.1
gpuid = -1


def main():
    holds = pickle.load(open(save_dir + '/holds.pkl', 'rb'))
    w2id = holds['vocab']
    id2w = {v: k for k, v in w2id.items()}
    classes = holds['classes']
    reversed_classes = {v: k for k, v in classes.items()}
    n_vocab = len(w2id)
    n_class = len(classes)

    left_encoder = nets.RNNEncoder(n_vocab, n_units, n_layer, dropout)
    right_encoder = nets.RNNEncoder(n_vocab, n_units, n_layer, dropout)
    model = nets.ContextClassifier(left_encoder, right_encoder, n_class)
    chainer.serializers.load_npz(save_dir + '/' + model_name, model)
    if gpuid >= 0:
        cuda.get_device(gpuid).use()
        model.to_gpu(gpuid)

    test, _, _ = make_dataset(test_path, w2id, classes)

    for i in range(0, len(test), batch_size):
        lxs, rxs, ts = seq_convert(test[i:i + batch_size], gpuid)
        predict_classes = model.predict(lxs, rxs, argmax=True)
        for i in range(len(lxs)):
            left_text = ''.join([id2w.get(idx, '') for idx in lxs[i]])
            right_text = ''.join([id2w.get(idx, '') for idx in rxs[i]])
            target = reversed_classes.get(ts[i][0])
            predict = reversed_classes.get(predict_classes[i])
            result = True if predict == target else False
            print('{} {}({}) {} {}'.format(left_text, predict, target, right_text, result))


if __name__ == '__main__':
    main()

