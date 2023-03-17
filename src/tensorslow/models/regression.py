from tensorslow.layers import Dense
from tensorslow.activations import Relu
from tensorslow.losses import L1Loss
from tensorslow.losses import L2Loss
from tensorslow.models import Model


in_dim = 1
x_1 = Dense(in_dim=in_dim, out_dim=5)
a_1 = Relu()
x_2 = Dense(in_dim=5, out_dim=1)
l1 = L1Loss()
l2 = L2Loss()


def ts_l1_regressor():
    return Model(x_1, a_1, x_2, l1)


def ts_l2_regressor():
    return Model(x_1, a_1, x_2, l2)