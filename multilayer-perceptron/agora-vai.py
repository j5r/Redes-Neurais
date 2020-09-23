import numpy as np

# counts the number of neurons from the input to the output layer
number_of_neurons_by_layer = [2, 3, 2, 4, 2]


def logistic(x: np.array):
    return 1/(1+np.exp(-x))


def dlogistic(x: np.array):
    return logistic(x)*(1-logistic(x))


# BEGIN ----------------------------------------------------------------
layers = []
for i in range(len(number_of_neurons_by_layer)):
    if i == 0:
        y = np.ones((number_of_neurons_by_layer[i]+1, 1))
        d = {"y": y}
        layers.append(d)
        continue
    w = np.random.normal(0, 1, (
        number_of_neurons_by_layer[i],
        number_of_neurons_by_layer[i-1]+1
    ))
    b = w[:, 0]
    y = np.ones((number_of_neurons_by_layer[i]+1, 1))
    v = y[1:, :]
    delta = np.zeros(v.shape)
    Delta_w = np.zeros(w.shape)
    d = {"weigths": w, "biases": b, "y": y,
         "v": v, "delta": delta, "Delta_w": Delta_w}
    layers.append(d)


def flow(input_):  # DONE
    layers[0]["y"][1:] = np.array(input_).flatten().reshape(len(input_), 1)
    for i_lay in range(1, len(layers)):
        layers[i_lay]["v"][:] = logistic(
            layers[i_lay]["weigths"] @ layers[i_lay-1]["y"]
        )


flow([1, 1])
_ = [print(i) for i in layers]
