from jrmlp import *
import numpy as np

# EXERCÍCIO 1 # ou-exclusivo
init([2, 2, 1])
IN = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUT = [[0], [1], [1], [0]]
train_batch(IN, OUT, plot=True, momentum=0.1, eta=1, maxit=1000)
save("ex1.py")


# EXERCÍCIO 2 #identidade 8
N = 8
init([N, np.math.ceil(np.math.log2(N)), N])
IN = np.eye(N).tolist()
OUT = IN
train_cyclic(IN, OUT, plot=True, momentum=0.1, eta=1, maxit=1000)
save("ex2.py")


# EXERCÍCIO 3 # identidade 15
N = 15
init([N, np.math.ceil(np.math.log2(N)), N])
IN = np.eye(N).tolist()
OUT = IN
train_cyclic(IN, OUT, plot=True, momentum=0.1, eta=1, maxit=1000)
save("ex3.py")


"""
# EXERCÍCIO 4
N = 50
init([N, np.math.ceil(np.math.log2(N)), N])
IN = np.eye(N).tolist()
OUT = IN

INtrain = IN[:40]
OUTtrain = OUT[:40]
INtest = IN[40:]
OUTtest = OUT[40:]
train_cyclic(INtrain, OUTtrain, plot=True, momentum=0.1, eta=1, maxit=10000)
e, m = test(INtest, OUTtest, plot=True)

save("ex_with_test.py")
"""
