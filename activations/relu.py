from tensorslow.activation import Activation


class Relu(Activation):
    op = lambda x: x if x > 0 else 0  
    def __init__(): 
        super.__init__(op)

    def backward(self, dout: Tensor) -> Tensor:
        dout = self.grad
        return dout 

