from tensorslow.linalg import Tensor
from tensorslow.activations import Activation

class Relu():
    f_x = lambda x: x if x > 0 else 0  
    df_x = lambda x: 1 if x > 0 else 0
    def __init__(): 
        super.__init__(fn)

    def backward(self, dout: Tensor) -> Tensor:
        self.grad = dout.unary_op(df_x)
        return dout 
