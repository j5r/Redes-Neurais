import numpy as np
from random import shuffle

__all__ = ["training", "testing"]


def sign(x):
    if x >= 0:
        return 1
    return -1


def Min(x, y):
    if x <= y:
        return 0
    return 1


def training(data: [[int, [int]]], **kw) -> dict:
    """
    data is sequence of pairs of type [int, [int]] as follows:
        [int=desired value, [int]=1d-array of data]
    kw["rate"] = rate of learning
    kw["tol"] = convergence tolerance
    kw["maxit"] = maximum of iterations allowed
    """
    kw.setdefault("rate", 0.9)
    kw.setdefault("tol", 1e-4)
    kw.setdefault("maxit", 1000)
    data = list(data)

    n = len(data)
    m = len(data[0][1])
    eps = kw["tol"]
    rate = kw["rate"]
    maxit = kw["maxit"]

    w = np.zeros(m+1, float)
    e = np.ones(n, float)
    j = -1
    theta = 0
    count = 0

    while count < maxit:
        count += 1
        j += 1
        if j == n:
            break

        temp = list(data[j][1])
        temp.append(-1)
        x = np.array(temp, float)
        e[j] = data[j][0] - sign(x@w)
        if abs(e[j]) < eps:
            continue
        w = w + rate * e[j] * x
        j = -1
        shuffle(data)
    kw["iterations_used"] = count
    kw["input_dimension"] = m
    kw["sample_size"] = n
    kw["weigths"] = w.tolist()

    # saving the solution
    f = open("solution.py", "w")
    f.write("sol = {")
    for k, v in kw.items():
        f.write(f'"{str(k)}": {str(v)},\n')
    f.write("}")
    f.close()
    return kw


def testing(data, **kw) -> dict:
    """
    data is sequence of 1d-array of data
    kw[1] = "className 1"
    kw[-1] = "className 2"
    """
    kw.setdefault("A", "Class A")
    kw.setdefault("B", "Class B")
    data = list(data)

    answer = []
    for data_item in data:
        name = data_item[0]
        data_item = data_item[1].tolist()
        data_item.append(-1)
        x = np.array(data_item)
        dot_product = x @ kw["weigths"]

        if abs(1 - dot_product) <= abs(-1 - dot_product):  # classe 1
            answer.append(
                {"name": name, "class": kw["A"],
                 "abs_error": abs(1 - dot_product)})
        else:  # classe -1
            answer.append(
                {"name": name, "class": kw["B"],
                 "abs_error": abs(-1 - dot_product)})
    kw["test"] = answer
    return kw
