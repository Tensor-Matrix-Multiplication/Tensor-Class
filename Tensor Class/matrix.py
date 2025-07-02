import collections.abc
import operator
import json
import numpy as np

class Tensor:
    def __init__(self, data):
        self._data = np.asarray(data)
        self._shape = self._data.shape
        self._rank = self._data.ndim

        
    @property
    def shape(self):
        """Returns the shape of the tensor"""
        return self._shape
    
    @property
    def rank(self):
        """Returns the rank (number of dimensions) of the tensor."""
        return self._rank
    
    def __repr__(self):
        return f"Tensor(data={self._data.__repr__()}, shape={self.shape})"
    
    def __str__(self):
        return str(self._data)
    
    #----Methods for Tensor and Component Wise Operations using numpy-----#    

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._data+other._data)
        elif isinstance(other, (int, float)):
            return Tensor(self._data + other)
        else:
            return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            # No explicit shape check needed; NumPy broadcasting handles it.
            return Tensor(self._data - other._data)
        elif isinstance(other, (int, float)):
            return Tensor(self._data - other)
        else:
            return NotImplemented

    def __rsub__(self, other):
        # For rsub, it's 'other - self', so we perform the operation in that order
        if isinstance(other, (int, float)):
            return Tensor(other - self._data)
        else:
            return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            # No explicit shape check needed; NumPy broadcasting handles it.
            return Tensor(self._data * other._data)
        elif isinstance(other, (int, float)):
            return Tensor(self._data * other)
        else:
            return NotImplemented
    
    def __rmul__(self, other):
        # __rmul__ can simply call __mul__ since multiplication is commutative
        return self.__mul__(other)
    
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            # Explicitly check for zero elements in the divisor Tensor
            if np.any(other._data == 0):
                raise ZeroDivisionError("Cannot divide by a Tensor containing zero elements.")
            return Tensor(self._data / other._data)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero scalar.")
            return Tensor(self._data / other)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            # Explicitly check for zero elements in the divisor Tensor
            if np.any(self._data == 0):
                raise ZeroDivisionError("Cannot divide by zero: Tensor contains zero elements.")
            return Tensor(other / self._data)
        else:
            return NotImplemented

    
    #---Kronecker Product for Tensors---#

    def kronecker_product(self, other):
        """
        Computes the Kronecker product of two tensors.
        This operation is generalized for N-dimensional arrays by NumPy's np.kron.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"Kronecker product can only be performed with another Tensor, "
                            f"received type {type(other)}")
        
        result_data = np.kron(self._data, other._data)
        
        return Tensor(result_data)


    def display(self):
        """
        Prints a nicely formatted representation of the tensor using numpy methods.
        """
        print(f"Tensor (shape={self.shape}, rank={self.rank}):")
        
        with np.printoptions(precision=4, suppress=True, threshold=1000):
            print(str(self._data))

    #---Save and Load the files using numpys inbuilt method---#
    def save(self, filepath):
        """
        Saves the tensor's data to a file using NumPy's .npy format.
        This format is efficient and preserves data types and shapes.
        """
        try:
            # Add .npy extension if not present for clarity
            if not filepath.endswith('.npy'):
                filepath += '.npy'
            np.save(filepath, self._data)
            print(f"Tensor successfully saved to {filepath}")
        except Exception as e: 
            print(f"Error saving tensor to {filepath}: {e}")

    @classmethod
    def load(cls, filepath):
        """
        Loads tensor data from a file (assumed to be .npy format)
        and returns a new Tensor instance.
        """
        try:
            if not filepath.endswith('.npy'):
                filepath += '.npy'
            data = np.load(filepath)
            print(f"Tensor successfully loaded from {filepath}")
            return cls(data)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except Exception as e: # Catch broader exceptions for loading
            print(f"Error loading tensor from {filepath}: {e}")
            return None

    


if __name__ == "__main__":
    # Define a constant for the file path prefix
    FILE_START = "file/"

    # Scalars
    t_scalar = Tensor(42)
    t_scalar.display()
    print(f"Type of internal data: {type(t_scalar._data)}")
    print(f"Scalar + 10: {(t_scalar + 10)._data}")
    print(f"10 - Scalar: {(10 - t_scalar)._data}\n")

    # Vectors
    t_vec1 = Tensor([1, 2, 3])
    t_vec2 = Tensor([4, 5, 6])
    t_vec1.display()
    print(f"Vector + Vector:\n{(t_vec1 + t_vec2)._data}")
    print(f"Vector - Vector:\n{(t_vec1 - t_vec2)._data}")
    print(f"Vector * 2:\n{(t_vec1 * 2)._data}\n")

    # Matrices
    t_mat1 = Tensor([[1, 2], [3, 4]])
    t_mat2 = Tensor([[5, 6], [7, 8]])
    t_mat1.display()
    t_mat2.display()
    print(f"Matrix + Matrix:\n{(t_mat1 + t_mat2)._data}")
    print(f"Matrix - Matrix:\n{(t_mat1 - t_mat2)._data}")
    print(f"Matrix / 2:\n{(t_mat1 / 2)._data}\n")

    # Rank-3 Tensors
    t_rank3_1 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    t_rank3_2 = Tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    t_rank3_1.display()
    print(f"Rank-3 + Rank-3:\n{(t_rank3_1 + t_rank3_2)._data}\n")
    print(f"Scalar + Rank-3:\n{(10 + t_rank3_1)._data}\n")
    print(f"Rank-3 * 0.5:\n{(t_rank3_1 * 0.5)._data}\n")


    print("\n--- Testing Kronecker Product (Generalized) ---")

    # Kronecker product of 2D matrices
    kp_m1 = Tensor([[1, 2], [3, 4]])
    kp_m2 = Tensor([[5, 6], [7, 8]])
    kp_result_matrix = kp_m1.kronecker_product(kp_m2)
    print(f"Kronecker Product (Matrix x Matrix):\n{kp_result_matrix._data}")
    print(f"Shape: {kp_result_matrix.shape}, Rank: {kp_result_matrix.rank}\n")

    # Kronecker product with a vector (1D)
    kp_vec1 = Tensor([1, 2, 3])
    kp_vec2 = Tensor([10, 100])
    kp_result_vec = kp_vec1.kronecker_product(kp_vec2)
    print(f"Kronecker Product (Vector x Vector):\n{kp_result_vec._data}")
    print(f"Shape: {kp_result_vec.shape}, Rank: {kp_result_vec.rank}\n")

    # Kronecker product of 3D tensors
    kp_3d_1 = Tensor([[[1, 0], [0, 1]]]) # Shape (1,2,2)
    kp_3d_2 = Tensor([[[5, 6], [7, 8]], [[9, 10], [11, 12]]]) # Shape (2,2,2)
    kp_result_3d = kp_3d_1.kronecker_product(kp_3d_2)
    print(f"Kronecker Product (3D x 3D):\n{kp_result_3d._data}")
    print(f"Shape: {kp_result_3d.shape}, Rank: {kp_result_3d.rank}\n")

    print("\n--- Testing save and load methods (NumPy .npy format) ---")
    file_path_npy = FILE_START+"my_tensor_numpy.npy"
    t_mat1.save(file_path_npy)

    loaded_tensor_npy = Tensor.load(file_path_npy)
    if loaded_tensor_npy:
        loaded_tensor_npy.display()
        print(f"Original tensor shape: {t_mat1.shape}, Loaded tensor shape: {loaded_tensor_npy.shape}")
        print(f"Original tensor data == Loaded tensor data: {np.array_equal(t_mat1._data, loaded_tensor_npy._data)}")

    # Test loading a non-existent file
    Tensor.load("non_existent_tensor_numpy.npy")

    # Test saving a rank-3 tensor
    t_rank3_1.save(FILE_START+"my_rank3_tensor_numpy.npy")
    loaded_rank3_tensor = Tensor.load(FILE_START+"my_rank3_tensor_numpy.npy")
    if loaded_rank3_tensor:
        loaded_rank3_tensor.display()

    print("\n--- Testing division by zero ---")
    t_zero = Tensor([[1, 2], [0, 4]])
    try:
        Tensor([[10, 10], [10, 10]]) / t_zero
    except ZeroDivisionError as e:
        print(f"Caught expected error: {e}")

    try:
        t_zero / 0
    except ZeroDivisionError as e:
        print(f"Caught expected error: {e}")

    try:
        5 / t_zero
    except ZeroDivisionError as e:
        print(f"Caught expected error: {e}")