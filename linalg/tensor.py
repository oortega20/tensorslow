import math
from typing import Union, List

class Tensor():
    def __init__(self, data: List[Union[int, float]], shape: List[int], init: str=''):
        self.shape = shape
        self.data = data
        self.init = init
        self.tensor = self._shape_tensor() if shape else []


    def __add__(self, t):
        return self.binary_op('+', t)


    def __sub__(self, t):
        return self.binary_op('-', t)


    def __mul__(self, t):
        return self.binary_op('*', t)


    def __truediv__(self, t):
        return self.binary_op('/', t)

    def __matmul__(self, t):
        return self.matmul(t)
 
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
            entry_num  = entry_num // elem        
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
        #TODO: implement this using the general rule
        return False
 
    def _set_entry(self, entry_loc, entry):
        def set_helper(tensor, entry_loc, entry):
            if len(entry_loc) == 1:
                tensor[entry_loc[0]] = entry
            else:
                set_helper(tensor[entry_loc[0]], entry_loc[1:], entry) 
        set_helper(self.tensor, entry_loc, entry)
        
    def _get_entry(self, entry_loc):
        def get_helper(tensor, entry_loc):
            if len(entry_loc) == 1:
                return tensor[entry_loc[0]]
            else:
                return get_helper(tensor[entry_loc[0]], entry_loc[1:]) 
        return get_helper(self.tensor, entry_loc)
 
    def _shape_tensor(self):
        def build_tensor(data, shape):
            entries = 0.0
            if self.init:
                if self.init == 'zeros':
                    entries = 0.0
                elif self.init == 'ones':
                    entries = 1.0
            return [build_tensor(data, shape[1:]) for i in range(shape[0])] if shape else entries
            
        self.tensor = build_tensor(self.data, self.shape)
        if self.data: 
            for i in range(self.num_entries):
                self._set_entry(self._entry_loc(i), self.data[i]) 
            
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
                s = self._get_entry(self.entry_loc(i, is_batch=True))
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
        tensor = Tensor([], self.shape)
        if (self._shape_compatible(tensor, 'binary') or 
            self._shape_broadcastable(tensor, 'binary')): 
            mod = min(self.num_entries, tensor.num_entries)
            total_entries = max(self.num_entries, tensor.num_entries)
            for i in range(total_entries):
                x = self._get_entry(self._entry_loc(i % mod))
                y = self._get_entry(self._entry_loc(i % mod)) 
                output = ops[op](x, y) 
                tensor._set_entry(self._entry_loc(i), output)   
        else:
            raise ValueError(f'''incompatible shapes for binary op: t1.shape {self.shape}, t2.shape {tensor2.shape}''')
        
        return tensor


    def matmul(self, tensor2):
        def _dot_product(tensor1, tensor2):        
            return sum(x * y for x, y in zip(tensor1, tensor2))
 
        def _matmul_helper(tensor1, tensor2):
            nonlocal current_batch
            nonlocal t
            x_dim, y_dim = t.shape[-2:]
            for x in range(x_dim):
                for y in range(y_dim):
                    p)
                    current_batch[x][y] = _dot_product(tensor1[x], tensor2[y])

        if self._shape_compatible(tensor2, 'matmul'):
            t = Tensor([], self.shape[:-2] + self.shape[-2:-1] + tensor2.shape[-1:])  
            tensor2 = tensor2.T
            if tensor2.order > 2:
                for i in range(math.prod(tensor2.shape[:-2])):
                    current_batch = t._get_entry(t.entry_loc(i, is_batch=True))
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
