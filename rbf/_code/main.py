"""
cross validation of the MLP

init([13, 15, 13, 3]) #model line #68
"""
import pandas as pd
import numpy as np
from random import shuffle
from jrmlp import (init,
                   train_cyclic,
                   train_batch,
                   test,
                   save)

print("Reading data...\t", end="")
# dataframe
df = pd.read_csv("wine.data", header=None)

# data access: df[header][row]

mx = df.max(axis=0)
for col in range(1, 14):
    df[col] /= mx[col]

class_1 = df[:][df[0] == 1].drop(columns=0).to_numpy().tolist()  # 59
class_2 = df[:][df[0] == 2].drop(columns=0).to_numpy().tolist()  # 59
class_3 = df[:][df[0] == 3].drop(columns=0).to_numpy().tolist()  # 48
print("Done.")
n_class1, n_class2, n_class3 = len(class_1), len(class_1), len(class_3)

shuffle(class_1)
shuffle(class_2)
shuffle(class_3)


output_1 = [1, 0, 0]
output_2 = [0, 1, 0]
output_3 = [0, 0, 1]


# Splitting data into folds
print("Folding data...\t", end="")
folds = []
stop = False
while not stop:
    new_fold = []
    for i in range(3):
        if class_1:
            new_fold.append([class_1.pop(), output_1.copy()])
        if class_2:
            new_fold.append([class_2.pop(), output_2.copy()])
        if class_3:
            new_fold.append([class_3.pop(), output_3.copy()])
        if not new_fold:
            stop = True
            break
    folds.append(new_fold)
print("Done.")
try:
    folds.remove([])
except:
    pass

shuffle(folds)

print("Initializing a battery of tests: Cross-Validation.")
print("Mode: STANDARD/CYCLIC")
init([13, 15, 13, 3])
errors = []
for turn in range(len(folds)):
    print(f"\vThe test fold is the \033[94m{turn}th\033[m.")
    test_fold = folds[turn]
    for fold_index in range(len(folds)):
        if fold_index == turn:
            continue
        print(f"Training with the \033[92m{fold_index}th\033[m fold.")
        training_fold = folds[fold_index]
        inputs = [tf[0] for tf in training_fold]
        outputs = [tf[1] for tf in training_fold]

        train_batch(inputs, outputs, plot=False,
                     eta=0.95, momentum=0.05, maxit=10000)
        inputs = [tf[0] for tf in test_fold]
        outputs = [tf[1] for tf in test_fold]
    errs, merrs = test(inputs, outputs)
    print(
        f"The mean quadratic error of the test fold is \033[95m{merrs}\033[m.")
    errors.append(merrs)
print("The battery of tests is done.")
print()
print("The mean erros of the r-folds were:")
print()
print(errors)
print()
errors = np.array(errors)
mean = np.mean(errors)
std = np.std(errors)
print(f"Its mean is {mean}.")
print(f"Its standard deviation is {std}.")
save("batch_wine_classifier.py")
print("THE END.")
