import collections.abc
import operator
import json
import numpy as np

class Tensor:
    def __init__(self, data):
        self._data = data
        self._shape = self._get_shape(data)
        self._rank = len(self._shape)

    
    def _get_shape(self, data):
        if not isinstance(data, collections.abc.Sequence):
            return () # Base case: Scalar

        if not data:
            raise ValueError("Cannot infer shape from empty list/sequence.")
        
        # Check if the data is correctly ordered
        if isinstance(data[0], collections.abc.Sequence):
            expected_len = len(data[0])
            for item in data:
                if isinstance(item, collections.abc.Sequence) and len(item) != expected_len:
                    raise ValueError("Input data is not rectangular (jagged array).")
        
        current_dimesion_size = len(data)

        sub_shape = self._get_shape(data[0])

        return (current_dimesion_size, ) + sub_shape
        
    @property
    def shape(self):
        """Returns the shape of the tensor"""
        return self._shape
    
    @property
    def rank(self):
        """Returns the rank (number of dimensions) of the tensor."""
        return self._rank
    
    def __repr__(self):
        return f"Tensor(data={self._data}, shape={self.shape})"
    
    def __str__(self):
        return str(self._data)
    
    #----Methods for Component Wise Operations-----#
    
    def _recursive_op(self, data1, data2_or_scalar, op_func):
        # Case 1: Both Scalar, Directly Evaluates
        if not isinstance(data1, collections.abc.Sequence) and \
            not isinstance(data2_or_scalar, collections.abc.Sequence):
            return op_func(data1, data2_or_scalar)
        
        # Case 2: data1 is a scalar and data2 is a sequence
        if not isinstance(data1, collections.abc.Sequence) and \
            isinstance(data2_or_scalar, collections.abc.Sequence):
                return [self._recursive_op(data1, item2, op_func) for item2 in data2_or_scalar]

        # Case 3: data1 is a sequence, data2_or_scalar is a scalar (Tensor op Scalar, e.g., Tensor + 10)
        if isinstance(data1, collections.abc.Sequence) and \
        not isinstance(data2_or_scalar, collections.abc.Sequence):
            return [self._recursive_op(item1, data2_or_scalar, op_func) for item1 in data1]

        # Case 4: Both are sequences (Tensor op Tensor) (Recursive Step)
        return [self._recursive_op(item1, item2, op_func) for item1, item2 in zip(data1, data2_or_scalar)]
    
    #----Element Wise Operations---#
    def _element_wise_op(self, other, op_func):
        """
        Helper method to prepare and execute an element-wise operation.
        Checks operand types and shapes, then delegates to _recursive_op.
        """
        if isinstance(other, Tensor):
            other_data_for_op = other._data
            if self.shape != other.shape:
                raise ValueError(f"Shapes must be identical for element-wise operation. "
                                f"Received {self.shape} and {other.shape}")
        elif isinstance(other, (int, float)):
            other_data_for_op = other # This is the scalar value itself
        else:
            raise TypeError(f"Unsupported operand type for element-wise operation: {type(other)}. "
                            f"Expected Tensor, int, or float.")

        return self._recursive_op(self._data, other_data_for_op, op_func)
    
    #---Dunder methods for element wise operations---#

    def __add__(self, other):
        result_data = self._element_wise_op(other, operator.add)
        return Tensor(result_data)
    
    def __sub__(self, other):
        result_data = self._element_wise_op(other, operator.sub)
        return Tensor(result_data)

    def __mul__(self, other):
        result_data = self._element_wise_op(other, operator.mul)
        return Tensor(result_data)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero scalar.")
        elif isinstance(other, Tensor):
            if self._contains_zero(other._data):
                raise ZeroDivisionError("Cannot divide by a Tensor containing zero elements.")
        
        result_data = self._element_wise_op(other, operator.truediv)
        return Tensor(result_data)
    
    def _contains_zero(self, data):
        """recursively checks if the element in the tensor has 0"""
        if not isinstance(data, collections.abc.Sequence):
            return data == 0
        for item in data:
            if self._contains_zero(item):
                return True
        return False

    #--- Right hand side (R-dunder) methods ---#
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            result_data = self._recursive_op(other, self._data, operator.sub)
            return Tensor(result_data)
        else:

            raise TypeError(f"Unsupported operand type for __rsub__ from Tensor: {type(other)}")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            if self._contains_zero(self._data):
                raise ZeroDivisionError("Cannot divide by zero: Tensor contains zero elements.")
            result_data = self._recursive_op(other, self._data, operator.truediv)
            return Tensor(result_data)
        else:
            raise TypeError(f"Unsupported operand type for __rtruediv__ from Tensor: {type(other)}")
    
    #---Kronecker Product for Rank-2 Tensors (Matrices)---#
    def kronecker_product(self, other):
        if self.rank != 2 or other.rank != 2:
            raise ValueError(f"Kronecker product (matrix definition) requires both tensors to be rank 2. "
                            f"Received ranks {self.rank} and {other.rank}.")
        
        m, n = self.shape
        p, q = other.shape

        result_rows = m * p
        result_cols = n * q

        result_matrix = [[0 for _ in range(result_cols)] for _ in range(result_rows)]

        for rs in range(m):
            for cs in range(n):

                rs_cs_val = self._data[rs][cs]
                rs_cs_block = self._recursive_op(rs_cs_val, other._data, operator.mul)

                block_start_row = rs * p
                block_start_col = cs * q

                for ro in range(p):
                    for co in range(q):
                        result_matrix[block_start_row+ro][block_start_col+co] = rs_cs_block[ro][co]
        
        return Tensor(result_matrix)

    def _format_data(self, data, indent_level):
        """
        Recursively formats the tensor data for display.
        """
        indent_str = "  " * indent_level
        if not isinstance(data, collections.abc.Sequence):
            return str(data)
        
        if not data: # Handle empty lists
            return "[]"

        # If it's a 1D list (vector)
        if not isinstance(data[0], collections.abc.Sequence):
            return "[ " + ", ".join(str(item) for item in data) + " ]"
        
        # For higher dimensions, format each sub-list on a new line
        formatted_sub_arrays = [self._format_data(sub_array, indent_level + 1) for sub_array in data]
        return "[\n" + ",\n".join(indent_str + "  " + sub_array for sub_array in formatted_sub_arrays) + "\n" + indent_str + "]"


    def display(self):
        """
        Prints a nicely formatted representation of the tensor.
        """
        print(f"Tensor (shape={self.shape}, rank={self.rank}):")
        if self.rank == 0:
            print(f"  {self._data}")
        else:
            print(self._format_data(self._data, 0))

    def save(self, filepath):
        """
        Saves the tensor's data to a file in JSON format.
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self._data, f)
            print(f"Tensor successfully saved to {filepath}")
        except IOError as e:
            print(f"Error saving tensor to {filepath}: {e}")

    @classmethod
    def load(cls, filepath):
        """
        Loads tensor data from a JSON file and returns a new Tensor instance.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"Tensor successfully loaded from {filepath}")
            return cls(data)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filepath}: {e}")
            return None
        except ValueError as e:
            print(f"Error creating Tensor from loaded data: {e}")
            return None

    


if __name__ == "__main__":
    # # Test Case 6: Rank-3 Tensor + Rank-3 Tensor
    # t3_1_data = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    # t3_2_data = Tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    # result_t3_add_t3 =  5 + t3_1_data 
    # print(f"t3_1 + t3_2 = {result_t3_add_t3}")
    # # Expected: [[[10, 12], [14, 16]], [[18, 20], [22, 24]]]

    # # Define matrices for Kronecker product testing
    # kp_m1 = Tensor([[1, 2], [3, 4]]) # 2x2
    # kp_m2 = Tensor([[5, 6], [7, 8]]) # 2x2
    
    # # Expected result for kp_m1 kron kp_m2 (4x4 matrix):
    # # [[ 5,  6, 10, 12],
    # #  [ 7,  8, 14, 16],
    # #  [15, 18, 20, 24],
    # #  [21, 24, 28, 32]]

    # kp_result_matrix = kp_m1.kronecker_product(kp_m2)
    # print(f"Matrix ([[1,2],[3,4]]) kron Matrix ([[5,6],[7,8]]):\n{kp_result_matrix}")
    # print(f"Shape: {kp_result_matrix.shape}, Rank: {kp_result_matrix.rank}\n") # Expected: (4,4), Rank: 2

    # t3_1_data.display()

    # t_matrix = Tensor([[2, 2, 3], [4, 5, 6], [7, 8, 9]])
    # t_matrix.display()

    # t_rank3 = Tensor([
    #     [[2, 2], [3, 4]],
    #     [[5, 6], [7, 8]],
    #     [[9, 10], [11, 12]]
    # ])
    # t_rank3.display()

    # print("\n--- Testing save and load methods ---")
    # file_path = "file/my_tensor.json"
    # t_matrix.save(file_path)

    # loaded_tensor = Tensor.load(file_path)
    # if loaded_tensor:
    #     loaded_tensor.display()
    #     print(f"Original tensor shape: {t_matrix.shape}, Loaded tensor shape: {loaded_tensor.shape}")
    #     print(f"Original tensor data == Loaded tensor data: {t_matrix._data == loaded_tensor._data}")


    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    b = np.array([1,0,0,1])
    b.reshape(1,1,4)
    b.reshape(1,4,1)
    
    print(b.T.shape)
    