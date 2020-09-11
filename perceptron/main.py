from mydata._input import *
from perceptron import *
import glob

trainment_file_names = glob.glob("./mydata/trainment/*.txt")
testing_file_names = glob.glob("./mydata/testing/*.txt")

# reading files
trainment_Items = [Item(i) for i in trainment_file_names]
testing_Items = [Item(i) for i in testing_file_names]

# getting data from files: [value, array]
array_trainment = [(i.value, i.array) for i in trainment_Items]
# getting data from files: [filename, array]
array_test = [(i.fname, i.array) for i in testing_Items]

try:  # attempting to import the solution already calculated
    from solution import sol as kw
    print("Solução foi importada com sucesso!")
    print("Não vamos treinar o modelo.")
except ImportError:
    print("Não foi encontrada uma solução.")
    print("Treinando modelo. . .")
    # getting optimal solution by training the model
    kw = training(array_trainment)
    print("Modelo treinado.")


kw["A"] = "Virado para cima"
kw["B"] = "Virado para baixo"

print()
print("Testando modelo. . .")
kw = testing(array_test, **kw)
for i, v in enumerate(kw["test"]):
    print("\t\033[94m", i+1, v)
    print(testing_Items[i])

print("\033[mTrabalho concluído.")
