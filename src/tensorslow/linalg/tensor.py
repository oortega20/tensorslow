import math
from iteration_utilities import deepflatten
from typing import Union, List, Tuple, Callable
from random import uniform


class Tensor:
    precision = 3

    def __init__(self, data: List[Union[int, float]], shape: Tuple[int], init: str = ''):
        self.shape = shape
        self.data = data
        self.init = init
        self.tensor = self._shape_tensor() if shape else []

    @staticmethod
    def op_error_msg(op, t1, t2):
        return f'unsupported operation: {op}({t1}, {t2})'

    def __add__(self, t):
        if isinstance(t, Tensor):
            return self.binary_op('+', t)
        elif isinstance(t, float) or isinstance(t, int):
            return self.unary_op(lambda x: x + t)
        else:
            raise TypeError(Tensor.op_error_msg('+', self, t))

    def __radd__(self, t):
        return self.__add__(t)

    def __sub__(self, t):
        if isinstance(t, Tensor):
            return self.binary_op('-', t)
        elif isinstance(t, float) or isinstance(t, int):
            return self.unary_op(lambda x: x - t)
        else:
            raise TypeError(Tensor.op_error_msg('-', self, t))

    def __rsub__(self, t):
        if isinstance(t, Tensor):
            return self.binary_op('-', t)
        elif isinstance(t, float) or isinstance(t, int):
            return self.unary_op(lambda x: t - x)
        else:
            raise TypeError(Tensor.op_error_msg('-', self, t))

    def __mul__(self, t):
        if isinstance(t, Tensor):
            return self.binary_op('*', t)
        elif isinstance(t, float) or isinstance(t, int):
            return self.unary_op(lambda x: x * t)
        else:
            raise TypeError(Tensor.op_error_msg('*', self, t))

    def __rmul__(self, t):
        return self.__mul__(t)

    def __truediv__(self, t):
        if isinstance(t, Tensor):
            return self.binary_op('/', t)
        elif isinstance(t, float) or isinstance(t, int):
            return self.unary_op(lambda x: x / t)
        else:
            raise TypeError(Tensor.op_error_msg('/', self, t))

    def __rtruediv__(self, t):
        if isinstance(t, Tensor):
            return self.binary_op('/', t)
        elif isinstance(t, float) or isinstance(t, int):
            return self.unary_op(lambda x: t / x)
        else:
            raise TypeError(Tensor.op_error_msg('/', self, t))

    def __matmul__(self, t):
        return self.matmul(t)

    def __pow__(self, t):
        if isinstance(t, int):
            result = Tensor([], init='ones', shape=self.shape)
            for _ in range(abs(t)):
                result = result * self
            return result if t >= 0 else 1 / result
        else:
            raise TypeError(Tensor.op_error_msg('**', self, t))

    def __rpow__(self, t):
        if isinstance(t, float) or isinstance(t, int):
            return self.unary_op(lambda x: t ** x)
        else:
            raise TypeError(Tensor.op_error_msg('**', self, t))

    def _print_helper(self, t, shape, levels, indent):
        precision = self.precision
        if len(shape) == 1:
            return '[' + ' '.join([f'{e:.{precision}f}' for e in t]) + ']'
        else:
            prefix, suffix = '[', ']'
            rep_str = ''
            for i, elem in enumerate(t):
                if i == 0:
                    rep_str += self._print_helper(elem, shape[1:], levels + 1, indent)
                else:
                    rep_str += ' ' * (levels + 1 + indent) + self._print_helper(elem, shape[1:], levels + 1, indent)
                if i < len(t) - 1:
                    rep_str += '\n'
            return prefix + rep_str + suffix

    def __repr__(self):
        return 'Tensor(' + self._print_helper(self.tensor, self.shape, 0, len('Tensor(')) + ')'

    def __str__(self):
        return self._print_helper(self.tensor, self.shape, 0, 0)

    @property
    def order(self):
        return len(self.shape)

    @property
    def num_entries(self):
        return math.prod(self.shape)

    def _entry_loc(self, entry_num, is_batch=False):
        entry_loc = []
        dims = self.shape[:-2][::-1] if is_batch else self.shape[::-1]
        for elem in dims:
            entry_loc = [entry_num % elem] + entry_loc
            entry_num = entry_num // elem
        return entry_loc

    def _shape_compatible(self, tensor, op: str) -> bool:
        if op == 'binary':
            return self.shape == tensor.shape
        elif op == 'matmul':
            self_output_dim = self.shape[-1]
            tensor_input_dim = tensor.shape[-2]
            return (self.order >= 2 and
                    self.order == tensor.order and
                    self.shape[:-2] == self.shape[:-2] and
                    self_output_dim == tensor_input_dim)
        return False

    def _shape_broadcastable(self, tensor):
        shape = self.shape
        broadcast_shape = tensor.shape
        if self.order < tensor.order:
            broadcast_shape = (1,) * (tensor.order - self.order) + self.shape
            shape = self.shape
        if tensor.order < self.order:
            broadcast_shape = (1,) * (self.order - tensor.order) + tensor.shape
            shape = self.shape
        for i in range(len(shape), -1):
            b_i, s_i = broadcast_shape[i], shape[i]
            if b_i != s_i and b_i != 1 and s_i != 1:
                return False
        return True

    def _set_entry(self, entry_loc, entry):
        def set_helper(tensor, loc, e):
            if len(loc) == 1:
                tensor[loc[0]] = e
            else:
                set_helper(tensor[loc[0]], loc[1:], e)

        set_helper(self.tensor, entry_loc, entry)

    def _get_entry(self, entry_loc):
        def get_helper(tensor, loc):
            if len(loc) == 1:
                return tensor[loc[0]]
            else:
                return get_helper(tensor[loc[0]], loc[1:])

        return get_helper(self.tensor, entry_loc)

    def _shape_tensor(self):
        def build_tensor(data, shape):
            entries = 0.0
            if self.init and self.init == 'ones':
                entries = 1.0
            return [build_tensor(data, shape[1:]) for _ in range(shape[0])] if shape else entries

        self.tensor = build_tensor(self.data, self.shape)
        if self.data:
            for i in range(self.num_entries):
                self._set_entry(self._entry_loc(i), self.data[i])

        if self.init and self.init == 'xavier':
            input_neurons = self.shape[0]
            lower, upper = -(1.0 / math.sqrt(input_neurons)), (1.0 / math.sqrt(input_neurons))
            data = [uniform(lower, upper) for _ in range(self.num_entries)]
            for i in range(self.num_entries):
                self._set_entry(self._entry_loc(i), data[i])
        if self.init and isinstance(self.init, Callable):
            data = [self.init() for _ in range(self.num_entries)]
            for i in range(self.num_entries):
                self._set_entry(self._entry_loc(i), data[i])
        return self.tensor

    @property
    def T(self):
        transposed_shape = self.shape[:-2] + self.shape[-2:][::-1]
        tensor = Tensor([], transposed_shape)
        if self.order == 2:
            x_dim, y_dim = self.shape
            for x in range(x_dim):
                for y in range(y_dim):
                    tensor.tensor[y][x] = self.tensor[x][y]
        else:
            for i in range(math.prod(self.shape[:-2])):
                t = tensor._get_entry(tensor._entry_loc(i, is_batch=True))
                s = self._get_entry(self._entry_loc(i, is_batch=True))
                x_dim, y_dim = self.shape[-2:]
                for x in range(x_dim):
                    for y in range(y_dim):
                        t[y][x] = s[x][y]
        return tensor

    def unary_op(self, op):
        tensor = Tensor([], self.shape)
        for i in range(self.num_entries):
            loc = self._entry_loc(i)
            x = self._get_entry(loc)
            tensor._set_entry(loc, op(x))
        return tensor

    def binary_op(self, op, tensor):
        ops = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
        }
        if (self._shape_compatible(tensor, 'binary') or
                self._shape_broadcastable(tensor)):
            order_diff = max(self.order, tensor.order) - min(self.order, tensor.order)
            if self.order < tensor.order:
                min_shape = order_diff * (1,) + self.shape
                max_shape = tensor.shape
            else:
                min_shape = order_diff * (1,) + tensor.shape
                max_shape = self.shape

            result_shape = tuple(map(max, zip(min_shape, max_shape)))
            result = Tensor([], result_shape)
            x_div, y_div = 1, 1
            for min_i, max_i in zip(min_shape, max_shape):
                if max_shape == self.shape:
                    y_div = max_i * y_div if max_i > min_i else y_div
                else:
                    x_div = max_i * x_div if max_i > min_i else x_div

            for i in range(result.num_entries):
                x_i = self._entry_loc(i // x_div)
                y_i = tensor._entry_loc(i // y_div)
                x = self._get_entry(x_i)
                y = tensor._get_entry(y_i)
                entry = ops[op](x, y)
                r_i = result._entry_loc(i)
                result._set_entry(r_i, entry)

            return result

    def matmul(self, tensor2):
        def _dot_product(tensor1, tensor2):
            return sum(x * y for x, y in zip(tensor1, tensor2))

        def _matmul_helper(tensor1, tensor2):
            nonlocal current_batch
            nonlocal t
            x_dim, y_dim = t.shape[-2:]
            for x in range(x_dim):
                for y in range(y_dim):
                    current_batch[x][y] = _dot_product(tensor1[x], tensor2[y])

        if self._shape_compatible(tensor2, 'matmul'):
            t = Tensor([], self.shape[:-2] + self.shape[-2:-1] + tensor2.shape[-1:])
            tensor2 = tensor2.T
            if tensor2.order > 2:
                for i in range(math.prod(tensor2.shape[:-2])):
                    current_batch = t._get_entry(t._entry_loc(i, is_batch=True))
                    t1 = self._get_entry(self._entry_loc(i, is_batch=True))
                    t2 = tensor2.get_entry(tensor2._entry_loc(i, is_batch=True))
                    _matmul_helper(t1, t2)
            else:
                current_batch = t.tensor
                _matmul_helper(self.tensor, tensor2.tensor)

            tensor2 = tensor2.T
            return t

        else:
            raise ValueError(f'''incompatible shapes for matmul: t1.shape {self.shape}, t2.shape {tensor2.shape}''')

    def _agg(self, method: str, axis=None):
        agg_ops = {
            'sum': sum,
            'max': max,
            'min': min,
            'mean': lambda x: sum(x) / len(x)
        }
        if not self.order == 2:
            raise NotImplementedError(
                f'''have only created aggregation for tensors of order 2: self.shape {self.shape}''')

        if axis is None:
            accum = list()
            for _ in range(self.num_entries):
                entry = self._get_entry(self._entry_loc(_))
                accum.append(entry)
            return agg_ops[method](accum)
        else:
            new_shape = self.shape[:axis] + self.shape[axis + 1:]
            result = Tensor([], new_shape)
            copy_tensor = self.T if axis == 0 else self
            x_dim, _ = copy_tensor.shape
            for x in range(x_dim):
                row = copy_tensor.tensor[x]
                result.tensor[x] = agg_ops[method](row)
            return result

    def sum(self, axis=None):
        return self._agg('sum', axis=axis)

    def max(self, axis=None):
        return self._agg('max', axis=axis)

    def min(self, axis=None):
        return self._agg('min', axis=axis)

    def mean(self, axis=None):
        return self._agg('mean', axis=axis)

    def expand_dims(self, axis=0):
        elems = list(deepflatten(self.tensor))
        if axis == 0:
            new_shape = (1,) + self.shape
        elif axis == self.order:
            new_shape = self.shape + (1,)
        elif axis < self.order:
            new_shape = self.shape[:axis] + (1,) + self.shape[axis:]
        else:
            raise ValueError(f'Invalid axis chosen for fn: expand_dims - axis: {axis}')
        return Tensor(elems, new_shape)

    def abs(self):
        return self.unary_op(lambda x: abs(x))

    def sqrt(self):
        return self.unary_op(lambda x: math.sqrt(x))

    @classmethod
    def diagflat(cls, data: list):
        t_shape = (len(data),) * 2
        new_data = [[0 for _ in range(len(data))] for _ in range(len(data))]
        for i in range(len(new_data)):
            new_data[i][i] = data[i]
        return Tensor(list(deepflatten(new_data)), t_shape)
