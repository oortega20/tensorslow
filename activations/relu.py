from tensorslow.linalg import Tensor

class Relu(activations.Activation):
    op = lambda x: x if x > 0 else 0  
    def __init__(): 
        super.__init__(op)

    def backward(self, dout: Tensor) -> Tensor:
        dout = self.grad
        return dout 
