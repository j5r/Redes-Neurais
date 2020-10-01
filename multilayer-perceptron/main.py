import numpy as np
from mlp import (
    # init(layers_size)
    init,
    # save(filename)
    save,
    # test(inputs, outputs, plot=False, tolerance=1e-4)
    test,
    # train_batch(inputs, outputs, eta=0.55, maxit=1000, momentum=0.1, plot=False)
    train_batch,
)


# EXERCÍCIO 1 # ou-exclusivo
init([2, 2, 1])
IN = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUT = [[0], [1], [1], [0]]
train_batch(IN, OUT, plot=True, momentum=0.01, eta=1, maxit=100000)
save("ex1.py")
test(IN, OUT)


# EXERCÍCIO 2 #identidade 8
N = 8
init([N, np.math.ceil(np.math.log2(N)), N])
IN = np.eye(N).tolist()
OUT = IN
train_batch(IN, OUT, plot=True, momentum=0.02, eta=1, maxit=100000)
save("ex2.py")
test(IN, OUT)


# EXERCÍCIO 3 # identidade 15
N = 15
init([N, np.math.ceil(np.math.log2(N)), N])
IN = np.eye(N).tolist()
OUT = IN
train_batch(IN, OUT, plot=True, momentum=0.02, eta=1, maxit=100000)
save("ex3.py")
test(IN, OUT)
