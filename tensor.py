import numpy as np
import pickle

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)

    def get_shape(self):
        return self.data.shape

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def is_zero(self):
        return np.all(self.data == 0)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return np.array_equal(self.data, other.data)

    def kron(self, other):
        return Tensor(np.kron(self.data, other.data))

    def __str__(self):
        return str(self.data)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])

    print("Tensor A:")
    print(A)
    print("Shape:", A.get_shape())

    print("\nTensor B:")
    print(B)

    print("\nA + B:")
    print(A + B)

    print("\nA * B:")
    print(A * B)

    print("\nKronecker Product A ⊗ B:")
    print(A.kron(B))

    print("\nIs A zero?", A.is_zero())

    print("\nSaving A to 'tensor_A.pkl'...")
    A.save("tensor_A.pkl")

    print("Loading into C...")
    C = Tensor.load("tensor_A.pkl")
    print("Loaded Tensor C:")
    print(C)