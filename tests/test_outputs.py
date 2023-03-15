import torch
import numpy as np
import tensorslow as ts

x = [[1.,2.,3.], [4.,5.,6.]]
x_n = np.array(x, dtype=np.float32)
x_ts = ts.linalg.tensor.Tensor(list(range(1,7)), (2,3))
x_t = torch.from_numpy(x_n)

w_1 = torch.ones(3, 3, requires_grad=True)
b_1 = torch.ones(2, 3, requires_grad=True)
w_2 = torch.ones(3, 3, requires_grad=True)
b_2 = torch.ones(2, 3, requires_grad=True)

def model(x):
    h_1 =  x @ w_1 + b_1
    print(h_1.shape, w_2.shape, b_2.shape)
    return h_1 @ w_2 + b_2 



pred_torch = model(x_t)
pred_torch.backward(x_t)
print(w_1.grad, 'torch gradw1')
print(w_2.grad, 'torch gradw2')
print(b_1.grad, 'torch gradb1')
print(b_2.grad, 'torch gradb2')

d1 = ts.layers.dense.Dense(in_shape=(2, 3), out_dims=3) 
d2 = ts.layers.dense.Dense(in_shape=(2, 3), out_dims=3)

def model_ts(x):
    h_1 = d1(x)
    h_2 = d2(h_1)
    return h_2

out = model_ts(x_ts)
print(out, 'out')
print(pred_torch, 'torch_out')

dx1 = d2.backward(x_ts)
dx0 = d1.backward(dx1) 

for k,v in d1.grad.items():
    if k != 'x':
        print(v, k, 'd1')

for k,v in d2.grad.items():
    if k != 'x':
        print(v, k, 'd2')
