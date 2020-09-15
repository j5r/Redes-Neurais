import numpy as np
from random import shuffle
learning_rate = 0.91
tolerance = 1e-2
global_tolerance = 1e-5
maxit = 1e3
mu = 0.1
global_maxit = 1e3


def sigmoidal(x):
    return 1/(1+np.exp(-x))


dataset = [
    [[1], [1, 1, 1]],
    [[0], [1, 1, 0]],
    [[0], [1, 0, 1]],
    [[1], [1, 0, 0]],
    [[0], [0, 1, 1]],
    [[1], [0, 1, 0]],
    [[1], [0, 0, 1]],
    [[0], [0, 0, 0]]
]

layers = np.array([2,  1])
inputdim = len(dataset[0][1])  # pattern dimension

# adding bias
layers += 1
layers[-1] -= 1
inputdim += 1

desired_output_set = [i[0] for i in dataset]
pattern_set = [np.array([1]+i[1]).reshape(len(i[1])+1, 1)
               for i in dataset]  # patterns
Errors = np.ones((len(dataset), 1))

w = [np.zeros((layers[0], inputdim))]
for i in range(len(layers[1:])):
    w.append(
        np.zeros((layers[i+1], layers[i]))
    )

delta = [list(range(len(layers))) for i in range(len(dataset))]
Delta = [list(range(len(layers))) for i in range(len(dataset))]
v = []
y = [pattern_set[0].copy()]
for i in range(len(layers)):
    v.append(
        np.zeros((layers[i], 1))
    )
    y.append(
        np.zeros((layers[i], 1))
    )


global_counter = 0

while True:
    global_tolerance *= 0.995
    mu *= 0.995
    global_counter += 1
    print("Contador global = {}, erro médio = {}\r".format(global_counter,
                                                           np.mean(Errors)), end="")
    # shuffle(dataset)
    for pattern in range(len(dataset)):
        count = 0
        y[0] = pattern_set[pattern].copy()
        d = np.array(desired_output_set[pattern]).reshape(
            (len(desired_output_set[pattern]), 1))
        while True:
            count += 1
            for i in range(len(layers)):
                v[i] = w[i] @ y[i]
                y[i+1] = sigmoidal(v[i])

            e = d - y[-1]
            E = e.T @ e
            Errors[pattern] = E
            if E < tolerance or count > maxit:
                break

            aux = np.diag(sigmoidal(v[-1]).flatten())
            delta[pattern][-1] = aux @ (np.eye(*aux.shape) - aux) @ e
            Delta[pattern][-1] = learning_rate * \
                delta[pattern][-1] @ y[-2].T + mu*Delta[pattern][-1]

            for i in range(len(layers)-2, -1, -1):
                aux = np.diag(sigmoidal(v[i]).flatten())
                delta[pattern][i] = aux @ (np.eye(*aux.shape) -
                                           aux) @ w[i+1].T @ delta[pattern][i+1]
                Delta[pattern][i] = learning_rate * \
                    delta[pattern][i] @ y[i].T + mu*Delta[pattern][i]

            for i in range(len(w)):
                w[i] += Delta[pattern][i]
    gradient_norm = np.array([np.mean(abs(Delta[pattern][i]))
                              for i in range(len(w)) for pattern in range(len(dataset))])
    if np.mean(gradient_norm) < global_tolerance or global_counter > global_maxit:
        print(f"\vGradient norm = {np.mean(gradient_norm)}.")
        break

print(w)
file = open("w.py", "w")
file.write("import numpy as np\n\nw = [")
for i in w:
    file.write("np." + i.__repr__())
    file.write(",\n")
file.write("]")
file.close()


######################################################
