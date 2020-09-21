import numpy as np
from random import shuffle


def linear(x: np.array):
    return x


def dlinear(x: np.array):
    return 1+0*x


def tanh(x: np.array):
    return np.tanh(x)


def dtanh(x: np.array):
    return 1-(np.tanh(x)**2)


def logistic(x: np.array):
    return 1/(1+np.exp(-x))


def dlogistic(x: np.array):
    return logistic(x)*(1-logistic(x))


class Layer:
    def __init__(self, n: int, next_=None, usebias=True, actf="logistic"):
        self.islast = (next_ is None)
        self.usebias = (usebias and not self.islast)
        self.rows = next_
        self.cols = n + int(bool(self.usebias))
        self.weigths = None
        self.biases = None
        self.next = next_
        self.my_n = n
        self.activation_function = actf
        if self.islast:
            self.activation_function = None

        if self.activation_function == "logistic":
            self.func = logistic
            self.dfunc = dlogistic
        elif self.activation_function == "tanh":
            self.func = tanh
            self.dfunc = dtanh
        elif self.activation_function == "linear":
            self.func = linear
            self.dfunc = dlinear
        elif self.activation_function is None:
            pass
        else:
            raise AttributeError(
                "\033[91mActivation function not recognized.\n" +
                "Set up some as 'logistic', 'linear' or 'tanh'.\033[m"
            )

    def initweigths(self):
        if self.cols and not self.islast:
            self.weigths = np.random.normal(
                0, 1,
                size=(self.rows, self.cols)
            )
        if self.usebias and not self.islast:
            self.biases = self.weigths[:, 0]

    def __repr__(self):
        s = "-"*30 + "\n"
        s += f"\033[1;94mself.rows =\033[0;32m {self.rows}\n"
        s += f"\033[1;94mself.cols =\033[0;32m {self.cols}\n"
        s += f"\033[1;94mself.islast =\033[0;32m {self.islast}\n"
        s += f"\033[1;94mself.usebias =\033[0;32m {self.usebias}\n"
        s += f"\033[1;94mself.weigths =\033[0;32m \n{self.weigths.__repr__()}"
        s += f"\n\033[1;94mself.biases =\033[0;32m \n{self.biases.__repr__()}\033[m"
        return s

    def __str__(self):
        s = f"\n\033[1;33mLayer(usebias={self.usebias}, my_n={self.my_n}, "
        s += f"lastlayer={self.islast}, next_={self.next})\n"
        s += f"\033[1;94mself.act_func =\033[0;32m \n{self.activation_function.__repr__()}\033[m\n"
        s += f"\033[1;94mself.biases = \033[0;32m \n{self.biases.__repr__()}\033[m\n"
        if not self.usebias:
            s += f"\033[1;94mself.weigths =\033[0;32m \n{self.weigths.__repr__()}\033[m\n"
        else:
            s += f"\033[1;94mself.weigths =\033[0;32m \n{self.weigths[:,1:].__repr__()}\033[m"
        return s


class Network:
    def __init__(self, layers: [Layer]):
        self.layers = layers
        self.y = [np.ones((lay.cols, 1))
                  if lay.usebias else np.zeros((lay.cols, 1))
                  for lay in self.layers]
        self.v = []
        for i, lay in enumerate(self.layers):
            self.v.append(self.y[i][int(lay.usebias):, :])

        for lay in self.layers:
            lay.initweigths()
        self.validate_weights_dimensions()

    def __repr__(self):
        s = ""
        for lay in self.layers:
            s += lay.__str__() + "\n" + "-"*30
        return s

    def validate_weights_dimensions(self):
        for i in range(1, len(self.layers)-1):
            if self.layers[i].my_n != self.layers[i-1].next:
                raise Exception(
                    f"\033[91mLayer {i-1}'s 'next_' do not match with 'n' of layer {i}.\033[m")

    def flow(self, input_: np.array):
        input_ = np.array(input_)
        try:
            if input_.size != self.v[0].size:
                raise IndexError(
                    f"\033[91mInput must have {self.v[0].size} elements.\033[m"
                )
        except AttributeError:
            raise Exception(
                f"\033[91mInput must be a numpy.array.\033[m"
            )

        self.v[0] = input_.reshape(self.v[0].shape)
        for i in range(1, len(self.layers)):
            self.v[i][:] = self.layers[i-1].func(
                self.layers[i-1].weigths @ self.y[i-1]
            )
        return self.y[-1].copy()

    def error(self, input_: np.array, desired: np.array):
        input_, desired = np.array(input_), np.array(desired)
        if desired.size != self.y[-1].size:
            raise IndexError(
                f"\033[91mDesired output must have {self.y[-1].size} elements.\033[m"
            )
        desired = desired.reshape(self.y[-1].shape)
        return desired - self.flow(input_)

    def error2(self, input_: np.array, desired: np.array):
        e = self.error(input_, desired)
        return e.T @ e


class MLP:
    def __init__(self, network: Network):
        self.net = network
        self.train_batch_error = []
        self.train_batch_gradient_norm = []
        self.train_cyclic_error = []

    def gradient(self, input_: list, output: list):
        inp = np.array(input_).flatten()
        out = np.array(output).flatten()
        error = self.net.error(inp, out)
        delta = [0*self.net.v[i].copy() for i in range(len(self.net.v))]
        del delta[0]
        delta[-1] = error * self.net.layers[-2].dfunc(self.net.v[-1])

        for i in range(len(self.net.layers)-3, -1, -1):
            delta[i] = self.net.layers[i].dfunc(
                self.net.v[i+1]
            ) * (self.net.layers[i+1].weigths[:, 1:].T @ delta[i+1])
        Delta_w = []
        for i in range(len(self.net.layers)-1):
            Delta_w.append(delta[i] @ self.net.y[i].T)
        return Delta_w

    def train_batch(self, inputs: [list], outputs: [list], eta=0.8, maxit=1000, momentum=0.3, tolerance=1e-4):
        inputs, outputs = [np.array(i).flatten() for i in inputs], [
            np.array(o).flatten() for o in outputs]
        ins_outs = list(zip(inputs, outputs))
        number_of_weigth_matrices = len(self.net.layers)-1

        self.train_batch_error.append(
            sum([self.net.error2(i, o) for i, o in ins_outs])[0]
        )
        print("Beginning to train as batch mode.")

        # initializing globalGradient
        globalGradient = [0 * self.net.layers[weigth].weigths
                          for weigth in range(number_of_weigth_matrices)]

        iteration_count = 0
        while iteration_count < maxit:
            iteration_count += 1
            delta_w_list = [self.gradient(i, o) for i, o in ins_outs]
            for weigth in range(number_of_weigth_matrices):
                globalGradient[weigth] = sum([eta * delta_w_list[pattern][weigth]
                                              for pattern in range(len(inputs))]) \
                    + momentum*globalGradient[weigth]
                self.net.layers[weigth].weigths += globalGradient[weigth]
            self.train_batch_error.append(
                sum([self.net.error2(i, o) for i, o in ins_outs])[0]
            )
            self.train_batch_gradient_norm.append(
                sum([np.linalg.norm(globalGradient[i]) for i in
                     range(number_of_weigth_matrices)])
            )
            if self.train_batch_gradient_norm[-1] < tolerance:
                break
        print("Traimnent is done.\n")

    def train_cyclic(self, inputs: [list], outputs: [list], eta=0.8, maxit=1000, momentum=0.3, tolerance=1e-4):
        inputs, outputs = [np.array(i).flatten() for i in inputs], [
            np.array(o).flatten() for o in outputs]
        ins_outs = list(zip(inputs, outputs))
        print("Beginning to train as cyclic mode.")
        iteration_count = 0
        delta_w_list_previous = [0*w for w in self.gradient(*ins_outs[0])]
        while iteration_count < maxit:
            iteration_count += 1
            shuffle(ins_outs)
            for i_pattern in range(len(ins_outs)):
                delta_w_list = self.gradient(*ins_outs[i_pattern])
                for i_w in range(len(delta_w_list)):
                    self.net.layers[i_w].weigths += eta * \
                        delta_w_list[i_w] + momentum*delta_w_list_previous[i_w]
                    delta_w_list_previous[i_w] = delta_w_list[i_w]
            self.train_cyclic_error.append(self.test_list(
                [i[0] for i in ins_outs], [i[1] for i in ins_outs])[0, 0])
            if self.train_cyclic_error[-1] < tolerance:
                break
        print("Trainment is done.\n")

    def test_one(self, input_: list, output: list):
        e = self.net.error(input_, output)
        return e.T @ e

    def test_list(self, inputs: [list], outputs: [list]):
        return sum([self.test_one(i, o) for i, o in zip(inputs, outputs)])

    def save(self):
        pass

    def __repr__(self):
        return self.net.__repr__()

    def __len__(self):
        return len(self.net.layers)-1
