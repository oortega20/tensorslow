from tensorslow.linalg import Tensor
from tensorslow.activations import Activation


f_x = lambda x: x if x > 0 else 0  
df_x = lambda x: 1 if x > 0 else 0

class Relu(Activation):
    def __init__(self): 
        super().__init__(f_x)

    def backward(self, dout: Tensor) -> Tensor:
        self.grad = self.x.unary_op(df_x)
        return self.grad * dout 
