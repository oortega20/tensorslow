from tensorslow.linalg import Tensor
from tensorslow.activations import Activation


f = lambda x: x if x > 0 else 0  
df = lambda x: 1 if x > 0 else 0

class Relu(Activation):
    def __init__(self): 
        super().__init__(f, df)

