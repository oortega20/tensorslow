import os


os.system('py -m pip install -e ../../tensorslow')
from tensorslow.activations import Softmax
from tensorslow.linalg import Tensor
from tensorslow.losses import CrossEntropyLoss
from tensorslow.datasets import MNIST
import numpy as np
from tqdm import tqdm


s_fn = Softmax()

m = MNIST()
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def softmax_grad(x): # Best implementation (VERY FAST)
    '''Returns the jacobian of the Softmax function for the given set of inputs.
    Inputs:
    x: should be a 2d array where the rows correspond to the samples
        and the columns correspond to the nodes.
    Returns: jacobian
    '''
    s = softmax(x)
    a = np.eye(s.shape[-1])
    temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
    temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
    temp1 = np.einsum('ij,jk->ijk',s,a)
    temp2 = np.einsum('ij,ik->ijk',s,s)
    return temp1-temp2


x_train, y_train = m.get_train_data()
for x, y in tqdm(zip(x_train[:3], y_train[:3]), desc='testing softmax', total=3):
    s_np = softmax(np.array(x.tensor))
    s_ts = np.array(s_fn(x).tensor)
    s = np.sum(np.abs(s_ts - s_np))
    if s > 1e-7:
        print('oh no, there is a bug')

    ds_np = softmax_grad(np.array(x.tensor))
    ds_ts = np.array(s_fn.backward().tensor)
    s = np.sum(np.abs(s_ts - s_np))
    if s > 1e-7:
        print('oh no there is a bug in backwards')
