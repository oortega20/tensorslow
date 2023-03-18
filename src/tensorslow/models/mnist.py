import dill
import pathlib

import tensorslow as ts
from tensorslow.layers import Dense
from tensorslow.activations import Relu
from tensorslow.activations import Softmax
from tensorslow.losses import CrossEntropyLoss
from tensorslow.models import Model


img_size = 28 * 28
x_1 = Dense('x_1', in_dim=img_size, out_dim=img_size)
a_1 = Relu()
x_2 = Dense('x_2', in_dim=img_size, out_dim=100)
a_2 = Relu()
x_3 = Dense('x_3', in_dim=100, out_dim=10)
s = Softmax()
ce = CrossEntropyLoss(units='bits')


def ts_mnist_classifier(from_ts=False):
    if from_ts:
        model_path = f'{ts.__path__[0]}/saved_models/ts_mnist_classifier.pkl'
        with open(pathlib.Path(model_path), 'rb') as f:
            model = dill.load(f)
        return model
    return Model(x_1, a_1, x_2, a_2, x_3, s, ce)
