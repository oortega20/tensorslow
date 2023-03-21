import dill
from abc import ABC
from typing import Tuple, Optional
from tabulate import tabulate, SEPARATING_LINE

from tensorslow.linalg import Tensor
from tensorslow.layers import Layer
from tensorslow.activations import Activation


loss = float


class Model(ABC):
    def __init__(self, *layers):
        self.layers = []
        for layer in layers:
            self.layers.append(layer)
        self.num_params = None

    def forward(self, x: Tensor, y: Optional[Tensor]=None):
        for layer in self.layers[:-1]:
            x = layer(x)

        if y:
            batch_loss, grad = self.layers[-1](x, y)
        else:
            return x
        return batch_loss, grad

    def __call__(self, x: Tensor, y: Optional[Tensor]) -> Tuple[loss, Tensor]:
        return self.forward(x, y=y)

    def backward(self, grad: Tensor) -> Tensor:
        for layer in self.layers[:-1][::-1]:
            grad = layer.backward(grad)
        return grad

    def save(self, path: str='model.pkl'):
        with open(path, 'wb') as f:
            dill.dump(self, f)

    def get_num_params(self):
        if not self.num_params:
            num_params = 0
            for layer in self.layers:
                if isinstance(layer, Layer):
                    for weight in layer.weights.values():
                        num_params += weight.num_entries
            self.num_params = num_params
        return self.num_params

    def summary(self):
        layer_name, layer_weights, layer_shapes, layer_params = None, None, None, None
        summaries = [['MODEL', '', '', self.get_num_params()]]
        for layer in self.layers[:-1]:
            if isinstance(layer, Layer):
                layer_name = layer.name
                layer_weights = tuple(layer.weights.keys())
                layer_shapes = tuple([l.shape for l in layer.weights.values()])
                layer_params = sum([l.num_entries for l in layer.weights.values()])
            elif isinstance(layer, Activation):
                layer_name = layer.__class__.__name__
                layer_weights = (None,)
                layer_shapes = (None,)
                layer_params = 0
            summaries.append([layer_name, layer_weights, layer_shapes, layer_params])
        print(tabulate(summaries, headers=['name', 'weight names', 'weight shapes', '#parameters'], tablefmt='github'))
