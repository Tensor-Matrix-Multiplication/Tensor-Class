import random
import csv

class Tensor:
    def __init__(self, dimensions, fill):

        self.shape = dimensions
        self.fill = fill
        self.dimension = len(dimensions)
        self.tensor = self.generate_tensor(self.shape)

#CREATING TENSOR

    def generate_tensor(self, shape, level=0):

        if len(shape) == 0:
            if self.fill == "random":
                return random.randint(0,9)
            elif self.fill == "custom":
                return int(input(f"Enter value at depth {level}"))
            elif isinstance(self.fill, (int, float)):
                return self.fill
            else:
                return 0
            
        else: return [self.generate_tensor(shape[1:], level + 1)for s in range(shape[0])]
    
    #SAVING TENSOR TO CSV

#TENSOR DISPLAY

    def __str__(self):
        return self.tensor_to_string(self.tensor)

    def tensor_to_string(self, tensor, level=0):
        if not isinstance(tensor, list):
            return str(tensor)

        indent = "  " * level
        result=""

        for i, sub_tensor in enumerate(tensor):
            result += f"{indent}Level {level} Index {i}:\n"
            result += self.tensor_to_string(sub_tensor, level + 1)
            result += "\n"
        return result
    
#ADDITION SCALAR AND COMPONENT

    def __add__(self, x):
        if isinstance(x, Tensor):
            if self.checkEquality(x):
                new_tensor = self.addComponent(self.tensor, x.tensor)
                t = Tensor(self.shape, "")
                t.tensor = new_tensor
                return t
            else:
                raise ValueError("Shapes don't match")

        elif isinstance(x, (int, float)):
            new_tensor = self.addScalar(self.tensor, x)
            t = Tensor(self.shape, "")
            t.tensor = new_tensor
            return t

        else:
            raise TypeError("Can only add a tensor or a scalar")
    
    def addComponent(self, t1, t2):
        if not isinstance(t1, list) and not isinstance(t2, list):
            return t1 + t2
        return [self.addComponent(a,b) for a, b in zip(t1, t2)]
    
    def addScalar(self, t1, s):
        if not isinstance(t1, list):
            return t1 + s
        return [self.addScalar(l, s) for l in t1]

#SUBTRACTION SCALAR AND COMPONENT

    def __sub__(self, x):
        if isinstance(x, Tensor):
            if self.checkEquality(x):
                new_tensor = self.subComponent(self.tensor, x.tensor)
                t = Tensor(self.shape, "")
                t.tensor = new_tensor
                return t
            else:
                raise ValueError("Shapes don't match")

        elif isinstance(x, (int, float)):
            new_tensor = self.subScalar(self.tensor, x)
            t = Tensor(self.shape, "")
            t.tensor = new_tensor
            return t

        else:
            raise TypeError("Can only subtract a tensor or a scalar")
    
    def subComponent(self, t1, t2):
        if not isinstance(t1, list) and not isinstance(t2, list):
            return t1 - t2
        return [self.subComponent(a,b) for a, b in zip(t1, t2)]
    
    def subScalar(self, t1, s):
        if not isinstance(t1, list):
            return t1 - s
        return [self.subScalar(l, s) for l in t1]

    def checkEquality(self, x):
        return self.shape == x.shape
    
#MULTIPLICATION SCALAR AND COMPONENT

    def __mul__(self, x):
        if isinstance(x, Tensor):
            if self.checkEquality(x):
                new_tensor = self.mulComponent(self.tensor, x.tensor)
                t = Tensor(self.shape, "")
                t.tensor = new_tensor
                return t
            else:
                raise ValueError("Shapes don't match")

        elif isinstance(x, (int, float)):
            new_tensor = self.mulScalar(self.tensor, x)
            t = Tensor(self.shape, "")
            t.tensor = new_tensor
            return t

        else:
            raise TypeError("Can only multiply a tensor or a scalar")
    
    def mulComponent(self, t1, t2):
        if not isinstance(t1, list) and not isinstance(t2, list):
            return t1 * t2
        return [self.mulComponent(a,b) for a, b in zip(t1, t2)]
    
    def mulScalar(self, t1, s):
        if not isinstance(t1, list):
            return t1 * s
        return [self.mulScalar(l, s) for l in t1]

    def checkEquality(self, x):
        return self.shape == x.shape
    
#DIVISION SCALAR AND COMPONENT

    def __truediv__(self, x):
        if isinstance(x, Tensor):
            if self.checkEquality(x):
                new_tensor = self.divComponent(self.tensor, x.tensor)
                t = Tensor(self.shape, "")
                t.tensor = new_tensor
                return t
            else:
                raise ValueError("Shapes don't match")

        elif isinstance(x, (int, float)):
            if x == 0:
                raise ZeroDivisionError("Can't divide by 0")
            new_tensor = self.divScalar(self.tensor, x)
            t = Tensor(self.shape, "")
            t.tensor = new_tensor
            return t

        else:
            raise TypeError("Can only divide by a tensor or a scalar")
    
    def divComponent(self, t1, t2):
        if not isinstance(t1, list) and not isinstance(t2, list):
            if t2 == 0:
                raise ZeroDivisionError("Can't divide by 0")
            return t1 / t2
        return [self.divComponent(a,b) for a, b in zip(t1, t2)]
    
    def divScalar(self, t1, s):
        if not isinstance(t1, list):
            return t1 / s
        return [self.divScalar(l, s) for l in t1]

#CHECK SIZE/ZERO  

    def checkEquality(self, x):
        return self.shape == x.shape
    
    def checkZero(self):
        def helper(t):
            if not isinstance(t, list):
                return t == 0
            return all(helper(sub) for sub in t)
        return helper(self.tensor)

#KRONECKERS PRODUCT AND HELPFUL METHODS
    @staticmethod
    def flatten(t):
        if not isinstance(t, list):
            return [t]
        flat = []
        for s in t:
            flat.extend(Tensor.flatten(s))
        return flat
    
    def get_shape(self):
       return self.shape
    
    @staticmethod
    def reshape(flat, shape):
        if not shape:
            return flat.pop(0)
        size = shape[0]
        return [Tensor.reshape(flat, shape[1:]) for _ in range(size)]

    
    def kronecker_product(self, t1):
        shape1 = self.shape
        shape2 = t1.get_shape()
    
        if len(shape1) != len(shape2):
            raise ValueError("Tensors must have the same number of dimensions")

        flat1 = Tensor.flatten(self.tensor)
        flat2 = Tensor.flatten(t1.tensor)

        result_flat = []
        for a in flat1:
            for b in flat2:
                result_flat.append(a * b)

        result_shape = [s1 * s2 for s1, s2 in zip(shape1, shape2)]
        reshaped = Tensor.reshape(result_flat[:], result_shape)

        t = Tensor(result_shape, fill="")  
        t.tensor = reshaped               
        return t



    
if __name__ == "__main__":
    first = Tensor([2,2,2], 3)    
    second = Tensor([2,2,2], 0) 
    third = first.kronecker_product(first)
    print(third)