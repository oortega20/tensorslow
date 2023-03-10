import math
from typing import Union, List

class Tensor():
    def __init__(self, data: List[Union[float|int]], shape: List[int], init: str=''):
        self.shape = shape
        self.data = data
        self.init = init
        self.tensor = self.shape_tensor() if shape else []
    
    @property
    def order(self):
      return len(self.shape)
        
    @property    
    def num_entries(self): 
        return math.prod(self.shape)
    
    @property
    def num_samples(self):
        order = 
        return math.prod(self.shape, start=2)
    
    
    def entry_loc(self, entry_num, is_batch=False):
        entry_loc = []
        dims = self.shape[:-2][::-1] if is_batch else self.shape[::-1]
        for elem in dims:
            entry_loc = [entry_num % elem] + entry_loc
            entry_num  = entry_num // elem        
        return entry_loc
    
    def shape_compatible(self, tensor: Tensor, op: str) -> bool:
        if op == 'add':
            return self.shape == tensor.shape
        elif op == 'matmul':
            
        else:
            return False
        return all(x == y for x,y in zipped)

    def shape_broadcastable(self, tensor):
        if tensor.order == 2 and self.shape[-1] == tensor2.shape[1] \
           and tensor2.shape[0] == 1:
            return self.shape[-1] == tensor.shape[1]
        elif len(tensor2.shape) == 1 and self.shape[-1] == tensor2.shape[0]:
            return True
        else:
            return False
 
    def set_entry(self, entry_loc, entry):
        def set_helper(tensor, entry_loc, entry):
            if len(entry_loc) == 1:
                tensor[entry_loc[0]] = entry
            else:
                set_helper(tensor[entry_loc[0]], entry_loc[1:], entry)
        
        set_helper(self.tensor, entry_loc, entry)
        
    def get_entry(self, entry_loc):
        def get_helper(tensor, entry_loc):
            if len(entry_loc) == 1:
                return tensor[entry_loc[0]]
            else:
                return get_helper(tensor[entry_loc[0]], entry_loc[1:])
        
        return get_helper(self.tensor, entry_loc)
 
    def shape_tensor(self):
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
        if len(self.shape) == 2:
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    tensor.tensor[y][x] = self.tensor[x][y]
        else:
            for i in range(self.num_entries(is_batch=True)):
                t = tensor.get_entry(tensor.entry_loc(i, is_batch=True))
                s = self.get_entry(self.entry_loc(i, is_batch=True))
                for x in range(self.shape[-2]):
                    for y in range(self.shape[-1]):
                        t[y][x] = s[x][y]
        return tensor
    
   
    def apply(self, op):
        sigmoid = lambda x: 1 / (1 + (math.e ** (-1 * x)))
        tanh = lambda x: ((math.e ** x) - math.e ** (-1 * x)) / ((math.e ** x) + math.e ** (-1 * x))
        relu = lambda x: x if x > 0 else 0
        tensor = Tensor([], self.shape)

        for i in range(self.num_entries()):
            x = self.get_entry(self.entry_loc(i))
            if op == 'sigmoid':
                tensor.set_entry(tensor.entry_loc(i), sigmoid(x))
            elif op == 'tanh':
                tensor.set_entry(tensor.entry_loc(i), tanh(x))
            else:
                tensor.set_entry(tensor.entry_loc(i), relu(x))

        return tensor
            
    
    def add(self, tensor2):
        
        tensor = Tensor([], self.shape)
        if self.shape_compatible(tensor2, 'add'):
            for i in range(self.num_entries()):
                total = self.get_entry(self.entry_loc(i)) + \
                        tensor2.get_entry(tensor.entry_loc(i))
                tensor.set_entry(self.entry_loc(i), total)   
        elif self.shape_broadcastable(tensor2):
            n = tensor2.shape[-1]
            for i in range(self.num_entries()):
                total = self.get_entry(self.entry_loc(i)) + \
                        tensor2.get_entry(tensor2.entry_loc(i % n))
                tensor.set_entry(tensor.entry_loc(i), total)
        else:
            raise ValueError(f'''incompatible shapes for add: t1.shape {self.shape}, t2.shape {tensor2.shape}''')
        
        return tensor
  
    def dot_product(self, tensor1, tensor2):        
        return sum(x + y for x, y in zip(tensor1, tensor2))

    
    def matmul(self, tensor2):   
        if self.shape_compatible(tensor2, 'matmul'):
            t = Tensor([], self.shape[:-2] + [self.shape[-2], tensor2.shape[-1]])
            tensor2 = tensor2.transpose
            
            def matmul_helper(tensor1, tensor2):
                nonlocal current_batch
                nonlocal t
                for x in range(t.shape[-2]):
                    for y in range(t.shape[-1]):
                        current_batch[x][y] = self.dot_product(tensor1[x], tensor2[y])
                
            if len(t.shape) > 2:
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


