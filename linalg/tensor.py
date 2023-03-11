import math
from typing import Union, List

class Tensor():
    def __init__(self, data: List[Union[float|int]], shape: List[int], init: str=''):
        self.shape = shape
        self.data = data
        self.init = init
        self.tensor = self._shape_tensor() if shape else []
    
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
    
    def _shape_compatible(self, tensor: Tensor, op: str) -> bool:
        if op == 'binary':
            return self.shape == tensor.shape
        elif op == 'matmul':
            self_output_dim = self.shape[-1]
            tensor_input_dim = tensor.shape[-2]
            return self.order >= 2 and
                   self.order == tensor.order and
                   self.shape[:-2] == self.shape[:-2] and
                   self_output_dim == tensor_input_dim
        return False

    def _shape_broadcastable(self, tensor):
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
            return [build_tensor(data, shape[1:]) for i in range(shape[0])] if shape else 0.0
            
        self.tensor = build_tensor(self.data, self.shape)
        
        if self.tensor:
            num_entries = min(len(self.data), self.num_entries())
            for i in range(self.num_entries()):
                self.set_entry(self.entry_loc(i), self.data[i]) 
        return self.tensor

        
    @property
    def transpose(self):
        transposed_shape = self.shape[:-2] + self.shape[-2::-1]
        tensor = Tensor([], transposed_shape)
        if self.order == 2:
            x_dim, y_dim = self.shape
            for x in range(x_dim):
                for y in range(y_dim):
                    tensor.tensor[y][x] = self.tensor[x][y]
        else:
            for i in range(self.num_entries(is_batch=True)):
                t = tensor.get_entry(tensor.entry_loc(i, is_batch=True))
                s = self.get_entry(self.entry_loc(i, is_batch=True))
                x_dim, y_dim = self.shape[-2:]
                for x in range(x_dim):
                    for y in range(y_dim):
                        t[y][x] = s[x][y]
        return tensor
    
   
    def _unary_op(self, op):
        tensor = Tensor([], self.shape)
        for i in range(self.num_entries):        
            loc = self.entry_loc(i)
            x = self.get_entry(loc)
            tensor.set_entry(loc, op(x))
        return tensor
            
    
    def _binary_op(self, tensor, op):
        ops = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
        }
        tensor = Tensor([], self.shape)
        if self.shape_compatible(tensor, 'binary') or 
           self.shape_broadcastable(tensor, 'binary'): 
            mod = min(self.entries, tensor.entries)
            total_entries = max(self.entries, tensor.entries)
            for i in range(total_entries):
                x = self.get_entry(self.entry_loc(i % mod))
                y = self.get_entry(self.entry_loc(i % mod)) 
                output = ops[op](x, y) 
                tensor.set_entry(self.entry_loc(i), output)   
        else:
            raise ValueError(f'''incompatible shapes for binary op: t1.shape {self.shape}, t2.shape {tensor2.shape}''')
        
        return tensor


    def matmul(self, tensor2):
        def dot_product(self, tensor1, tensor2):        
            return sum(x + y for x, y in zip(tensor1, tensor2))
 
        def matmul_helper(tensor1, tensor2):
            nonlocal current_batch
            nonlocal t
            x_dim, y_dim = t.shape[-2:]
            for x in range(x_dim):
                for y in range(y_dim):
                    current_batch[x][y] = self.dot_product(tensor1[x], tensor2[y])
              
        if self.shape_compatible(tensor2, 'matmul'):
            t = Tensor([], self.shape[:-2] + [self.shape[-2], tensor2.shape[-1]])
            tensor2 = tensor2.transpose
  
            if tensor2.order > 2:
                for i in range(t.num_entries(is_batch=True)):
                    current_batch = t.get_entry(t.entry_loc(i, is_batch=True))
                    t1 = self.get_entry(self.entry_loc(i, is_batch=True))
                    t2 = tensor2.get_entry(tensor2.entry_loc(i, is_batch=True))
                    matmul_helper(t1, t2)
            else:
                current_batch = t.tensor
                matmul_helper(self.tensor, tensor2.tensor)
                
            tensor2 = tensor2.transpose  
            return t
   
        else:
            raise ValueError(f'''incompatible shapes for matmul: t1.shape {self.shape}, t2.shape {tensor2.shape}'


