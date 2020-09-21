import numpy as np
import matplotlib.pyplot as plt
from mlp import (MLP, Network, Layer)

mlp = MLP(
    Network([
        Layer(2, next_=2),
        Layer(2, next_=1),
        Layer(1)
    ])
)

IN = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUT = [[0], [1], [1], [0]]

DICT = {"inputs": IN, "outputs": OUT, "momentum": 1e-2, "eta": .9,
        "tolerance": 1e-4, "maxit": 10000}

print(mlp.test_list(IN, OUT))
# mlp.train_cyclic(**DICT)
#print(mlp.test_list(IN, OUT))
mlp.train_batch(**DICT)
print(mlp.test_list(IN, OUT))
print(mlp)

"""
plt.semilogy(list(range(len(mlp.train_cyclic_error))),
             mlp.train_cyclic_error, 'b*-')
plt.xlabel("iteration")
plt.ylabel("Quadratic error sum")
plt.title("ERROR")
plt.grid()

plt.show()
"""
