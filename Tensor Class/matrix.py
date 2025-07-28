import collections.abc
import operator
import json
import numpy as np
import os
from pysat.solvers import Glucose3  # We'll use the Glucose3 solver
from pysat.formula import CNF

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
            return Tensor((self._data+other._data)%2)
        elif isinstance(other, (int, float)):
            return Tensor((self._data + other)%2)
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
        """
        try:
            # Add .npy extension if not present for clarity
            if not filepath.endswith('.npy'):
                filepath += '.npy'

            directory = os.path.dirname(filepath)
            # Create the directory if it doesn't exist
            if directory: # Ensure directory is not an empty string
                os.makedirs(directory, exist_ok=True)

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
    
    def reshape(self, *new_shape):
        """
        Reshapes the tensor to the given new_shape.
        The total number of elements must remain the same.
        Returns a new Tensor instance with the reshaped data.
        """
        try:
            reshaped_data = self._data.reshape(new_shape)
            return Tensor(reshaped_data)
        except ValueError as e:
            raise ValueError(f"Cannot reshape tensor with current shape {self.shape} to {new_shape}: {e}")
        
    def reshape_inplace(self, *new_shape):
        """
        Reshapes the tensor to the given new_shape IN-PLACE.
        The total number of elements must remain the same.
        Updates the tensor's shape and rank properties.
        """
        try:
            self._data = self._data.reshape(new_shape)
            self._shape = self._data.shape
            self._rank = self._data.ndim
        except ValueError as e:
            raise ValueError(f"Cannot reshape tensor with current shape {self.shape} to {new_shape}: {e}")



if __name__ == "__main__":
    # Define a constant for the file path prefix
    FILE_START = "public/"
    
    # --- Step 1: Create Encoded Basis Tensors ---
    def create_encoded_basis(index, shape):
        """Creates a vector of length 7, reshapes it, and returns it as a Tensor."""
        vec = np.zeros(4, dtype=int)
        vec[index] = 1
        return Tensor(vec.reshape(shape))

    # Reshape basis vectors to create the desired (c, a, b) axis order in the final tensor.
    a = [create_encoded_basis(i, (1, 4, 1)) for i in range(4)]
    b = [create_encoded_basis(i, (1, 1, 4)) for i in range(4)]
    c = [create_encoded_basis(i, (4, 1, 1)) for i in range(4)]
    
    def to_idx(r, c):
        return (r - 1) * 2 + (c - 1)

    print("--- Verification of a 7-Multiplication Scheme (Encoded Vector Method) ---")
    print("This script uses Z_2 arithmetic (modulo 2) as required by the scheme.\n")

    # --- Step 2: Construct each of the 7 summands (M_l) ---
    # The product order is C.kron(A).kron(B) to get the desired (c,a,b) axes
    
    # M1 = (
    # c[to_idx(1,1)] + c[to_idx(1,2)] + c[to_idx(2,1)] + c[to_idx(2,2)]
    # ).kronecker_product(
    #     a[to_idx(1,2)] + a[to_idx(2,1)] + a[to_idx(2,2)]
    # ).kronecker_product(
    #     b[to_idx(1,1)] + b[to_idx(1,2)] + b[to_idx(2,1)] + b[to_idx(2,2)]
    # )

    # M2 = (
    #     c[to_idx(1,2)]
    # ).kronecker_product(
    #     a[to_idx(1,1)]
    # ).kronecker_product(
    #     b[to_idx(1,2)] + b[to_idx(2,1)] + b[to_idx(2,2)]
    # )

    # M3 = (
    #     c[to_idx(1,1)] + c[to_idx(1,2)] + c[to_idx(2,1)] + c[to_idx(2,2)]
    # ).kronecker_product(
    #     a[to_idx(1,2)] + a[to_idx(2,1)]
    # ).kronecker_product(
    #     b[to_idx(1,1)] + b[to_idx(2,1)] + b[to_idx(2,2)]
    # )

    # M4 = (
    #     c[to_idx(1,1)] + c[to_idx(1,2)] + c[to_idx(2,1)] + c[to_idx(2,2)]
    # ).kronecker_product(
    #     a[to_idx(2,2)]
    # ).kronecker_product(
    #     b[to_idx(1,1)] + b[to_idx(1,2)] + b[to_idx(2,1)] + b[to_idx(2,2)]
    # )

    # M5 = (
    #     c[to_idx(1,1)] + c[to_idx(1,2)] + c[to_idx(2,1)] + c[to_idx(2,2)]
    # ).kronecker_product(
    #     a[to_idx(1,2)] + a[to_idx(2,1)]
    # ).kronecker_product(
    #     b[to_idx(1,1)] + b[to_idx(1,2)] + b[to_idx(2,1)] + b[to_idx(2,2)]
    # )

    # M6 = (
    #     c[to_idx(1,2)]
    # ).kronecker_product(
    #     a[to_idx(1,1)]
    # ).kronecker_product(
    #     b[to_idx(1,2)] + b[to_idx(2,1)] + b[to_idx(2,2)]
    # )

    # M7 = (
    #     c[to_idx(1,1)] + c[to_idx(1,2)] + c[to_idx(2,1)] + c[to_idx(2,2)]
    # ).kronecker_product(
    #     a[to_idx(1,2)] + a[to_idx(2,1)]
    # ).kronecker_product(
    #     b[to_idx(1,1)] + b[to_idx(2,1)] + b[to_idx(2,2)]
    # )
    M1 = (c[to_idx(1,1)] + c[to_idx(1,2)] + c[to_idx(2,1)] + c[to_idx(2,2)]).kronecker_product(
            a[to_idx(1,1)] + a[to_idx(2,2)]
        ).kronecker_product(
            b[to_idx(1,1)] + b[to_idx(2,2)]
        )

    M2 = (c[to_idx(2,1)] + c[to_idx(2,2)]).kronecker_product(
            a[to_idx(2,1)] + a[to_idx(2,2)]
        ).kronecker_product(
            b[to_idx(1,1)]
        )

    M3 = (c[to_idx(1,2)] + c[to_idx(2,2)]).kronecker_product(
            a[to_idx(1,1)]
        ).kronecker_product(
            b[to_idx(1,2)] - b[to_idx(2,2)]
        )

    M4 = (c[to_idx(1,1)] + c[to_idx(2,1)]).kronecker_product(
            a[to_idx(2,2)]
        ).kronecker_product(
            b[to_idx(2,1)] - b[to_idx(1,1)]
        )

    M5 = (c[to_idx(1,1)] + c[to_idx(1,2)]).kronecker_product(
            a[to_idx(1,1)] + a[to_idx(1,2)]
        ).kronecker_product(
            b[to_idx(2,2)]
        )

    M6 = (c[to_idx(2,2)]).kronecker_product(
            a[to_idx(2,1)] - a[to_idx(1,1)]
        ).kronecker_product(
            b[to_idx(1,1)] + b[to_idx(1,2)]
        )

    M7 = (c[to_idx(1,1)]).kronecker_product(
            a[to_idx(1,2)] - a[to_idx(2,2)]
        ).kronecker_product(
            b[to_idx(2,1)] + b[to_idx(2,2)]
        )


    

    # --- Step 3: Sum all 23 tensors ---
    Result = M1 + M2 + M3 + M4 + M5 + M6 + M7 
    
    print("\n--- Final Resulting Tensor (Corrected) ---")
    Result.display()
