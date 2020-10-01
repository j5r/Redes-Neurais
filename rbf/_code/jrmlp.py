import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

# INITIALIZING GLOBAL VARIABLES
global layers
global ERROR
global number_of_neurons_by_layer
layers = []
ERROR = []


def logistic(x: np.array):
    return 1/(1+np.exp(-x))


def dlogistic(x: np.array):
    return logistic(x)*(1-logistic(x))


def init(nneurons):
    global number_of_neurons_by_layer
    number_of_neurons_by_layer = nneurons
    """
    "number_of_neurons_by_layer" is a list that counts the number of neurons
    from the input to the output layer
    "y" is the concatenation of "1" and "v", the first layer has only this parameter;
    "v" is the flow vector;
    "weigths" are the concatenation of a column of biases followed by columns of weigths;
    "biases" are the first column of "weigths";
    "delta" is the lower-case delta from class notes (derivative of squared error with
    respect to "v");
    "Delta_w" is the upper-case Delta from class notes (the step for updating the
    weigths: it need one more parameter, the length-step of gradient descent method,
    the parameter <eta> from class notes);
    "error" is the error vector between the desired and the obtained outputs, it is
    stored only at the last layer;
    "error2" is the summation of squared errors calculated at the vector "error", it
    is also stored only at the last layer;
    """
    layers.clear()
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
    layers[-1]["error"] = 0*layers[-1]["y"][1:, :].copy()
    layers[-1]["error2"] = 0
# END: INITIALIZING THE LAYERS


def flow(input_):
    """
    it makes the flow of a given input through the network,
    all data are stored in the layers "y" and "v"
    """
    global number_of_neurons_by_layer
    if len(input_) != number_of_neurons_by_layer[0]:
        raise IndexError(
            f"\033[91mInput length is incorrect. It must be {number_of_neurons_by_layer[0]}.\033[m")
    layers[0]["y"][1:] = np.array(input_).flatten().reshape(len(input_), 1)
    for i_lay in range(1, len(layers)):
        layers[i_lay]["v"][:] = logistic(
            layers[i_lay]["weigths"] @ layers[i_lay-1]["y"]
        )


def error(input_, output):
    """
    it computes the error vector between desired and obtained output,
    stored at the last layer
    """
    global number_of_neurons_by_layer
    if len(output) != number_of_neurons_by_layer[-1]:
        raise IndexError(
            f"\033[91mDesired output length is incorrect. It must be {number_of_neurons_by_layer[-1]}.\033[m")
    output = np.array(output).reshape(len(output), 1)
    flow(input_)
    layers[-1]["error"] = output - layers[-1]["v"]


def error2(input_, output):
    """
    it computes the sum of quadratic error of a given input,
    stored at the last layer
    """
    error(input_, output)
    layers[-1]["error2"] = 0.5*layers[-1]["error"].T @ layers[-1]["error"]
    return layers[-1]["error2"].copy().item()


def backpropagate(eta, momentum):
    """
    it computes "delta" and "Delta_w"
    """
    for i_lay in range(len(layers)-1, 0, -1):
        lay = layers[i_lay]
        if i_lay == len(layers)-1:
            lay["delta"] = lay["error"] * dlogistic(lay["v"])
        else:
            lay["delta"] = (layers[i_lay+1]["weigths"][:, 1:].T  @ layers[i_lay+1]
                            ["delta"]) * dlogistic(lay["v"])
        lay["Delta_w"] = eta * lay["delta"] @ layers[i_lay - 1]["y"].T +\
            momentum * lay["Delta_w"]


def updateweigths():
    """
    once you have "Delta_w", it makes $ w <- w + Delta_w
    """
    for i_lay in range(1, len(layers)):
        layers[i_lay]["weigths"] += layers[i_lay]["Delta_w"]


def getweigths():
    """
    it gets the list of weigths
    """
    ls = []
    for i_lay in range(1, len(layers)):
        ls.append(layers[i_lay]["weigths"])
    return ls


def get_Delta_weigths():
    """
    it gets the list of "Delta_w"
    """
    ls = []
    for i_lay in range(1, len(layers)):
        ls.append(layers[i_lay]["Delta_w"].copy())
    return ls


def set_Delta_w_to_zero():
    for i_lay in range(1, len(layers)):
        layers[i_lay]["Delta_w"] *= 0


def setweigths(ls):
    """
    it sets the list of "weigths": they must be of the same type of data (numpy.array)
    and same dimension.
    """
    for i_lay in range(1, len(layers)):
        layers[i_lay]["weigths"] = ls[i_lay-1]


def set_Delta_weigths(ls):
    """
    it sets the list of "Delta_w": they must be of the same type of data (numpy.array)
    and same dimension.
    """
    for i_lay in range(1, len(layers)):
        layers[i_lay]["Delta_w"] = ls[i_lay-1]


def train_cyclic(inputs, outputs, eta=0.55, maxit=1000, momentum=0.1, plot=False, init_momentum=False):
    """
    it performs the cyclic mode of training
    """
    global ERROR
    ERROR.clear()
    if init_momentum:
        set_Delta_w_to_zero()
    min_error = 100
    ins_outs = list(zip(inputs, outputs))
    counter = 0
    while counter <= maxit:
        eta *= 0.9995
        counter += 1
        shuffle(ins_outs)
        for pair in ins_outs:
            i, o = pair
            error2(i, o)
            ERROR.append(layers[-1]["error2"].item())
            try:
                if ERROR[-1] < min_error:
                    min_error = ERROR[-1]
                    optimal_w = getweigths()
                    min_error_counter = counter
                    print(
                        "At counter = {:5d}, the minimum error is {:.5g}".format(min_error_counter, min_error), end="\r")
            except:
                pass
            backpropagate(eta, momentum)
            updateweigths()
    setweigths(optimal_w)
    print(f"\vMinimum error reached at the {min_error_counter}th cycle")
    if plot:
        if len(ERROR) > 2000:
            l = int(np.ceil(len(ERROR)/1000))
            ERROR = ERROR[0:-1:l]
        plt.plot(np.arange(len(ERROR)), ERROR, "b*-")
        plt.xlabel("Number of cycles")
        plt.ylabel("Sum of quadratic errors")
        plt.title("ERROR vs CYCLES")
        plt.grid()
        plt.show()


def train_batch(inputs, outputs, eta=0.55, maxit=1000, momentum=0.1, plot=False, init_momentum=False):
    """
    it performs the batch mode of training
    """
    global ERROR
    ERROR.clear()
    if init_momentum:
        set_Delta_w_to_zero()
    min_error = 100
    ins_outs = list(zip(inputs, outputs))
    counter = 0
    while counter <= maxit:
        eta *= 0.9995
        counter += 1
        shuffle(ins_outs)
        Dws = []
        errors = []
        for pair in ins_outs:
            i, o = pair
            error2(i, o)
            errors.append(layers[-1]["error2"].item())
            ws = getweigths()
            backpropagate(eta, momentum)
            Dws.append(get_Delta_weigths())
            setweigths(ws)
        ERROR.append(sum(errors))
        try:
            if ERROR[-1] < min_error:
                min_error = ERROR[-1]
                optimal_w = getweigths()
                min_error_counter = counter
                print(
                    "At counter = {:5d}, the minimum error is {:.5g}".format(min_error_counter, min_error), end="\r")
        except:
            pass
        Delta_w = []
        for ws in range(len(Dws[0])):
            Delta_w.append(
                sum(
                    [Dws[pattern][ws] for pattern in range(len(ins_outs))]
                )
            )
        set_Delta_weigths(Delta_w)
        updateweigths()
    setweigths(optimal_w)
    print(f"\vMinimum error reached at the {min_error_counter}th cycle")
    if plot:
        if len(ERROR) > 2000:
            l = int(np.ceil(len(ERROR)/1000))
            ERROR = ERROR[0:-1:l]
        plt.plot(np.arange(len(ERROR)), ERROR, "b*-")
        plt.xlabel("Number of cycles")
        plt.ylabel("Sum of quadratic errors")
        plt.title("ERROR vs CYCLES")
        plt.grid()
        plt.show()


def test(inputs, outputs, plot=False):
    ins_outs = list(zip(inputs, outputs))
    errors = []
    for io in ins_outs:
        i, o = io
        error2(i, o)
        errors.append(layers[-1]["error2"].item())
    mean_error = sum(errors)/len(errors)
    if plot:
        if len(errors) > 2000:
            l = int(np.ceil(len(errors)/1000))
            errors = errors[0:-1:l]
        plt.plot(np.arange(len(errors)), errors, "b*-")
        plt.xlabel("pattern to test")
        plt.ylabel("Sum of quadratic errors")
        plt.title("ERROR vs PATTERN")
        plt.grid()
        plt.show()

    print(f"TEST SAYS: Mean square error is {mean_error}.")
    return errors, mean_error


def save(filename):
    W = getweigths()
    file = open(filename, "w")
    file.write("import numpy as np\n\n")
    file.write("weigths = [\n")
    for w in W:
        file.write("np.")
        file.write(w.__repr__()+",\n")
    file.write("]\n\n")
    file.close()
