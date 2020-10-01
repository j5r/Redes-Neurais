# Trabalho da disciplina de Redes Neurais Artificiais

## Grupo nº 6, com os integrantes

*  Júlio César de Melo Cândido NUSP 11926153

*  Junior Rodrigues Ribeiro NUSP 9725190

*  Luiz Henrique Romero NUSP 10095090


# DESCRIÇÃO

Este exercício implementa o algoritmo para Rede Perceptron Multicamada

1. resolver o problema XOR com 2 neurônios na camada única oculta, com entradas de dimensão $n=2$ e saídas de dimensão $m=1$.

2. resolver o "auto-associador" com $\log_2(n)$ neurônios na única camada oculta, com entradas de dimensão $n=8$ e saídas de dimensão $m=15$.

2. resolver o "auto-associador" com $\log_2(n)$ neurônios na única camada oculta, com entradas de dimensão $n=15$ e saídas de dimensão $m=15$.


# REQUERIMENTOS

O programa foi escrito na linguagem Python3, e faz uso da biblioteca `numpy`, a qual pode ser facilmente instalada com `pip3 install numpy`.

# EXECUÇÃO

Para executar o programa, abra um terminal com o comando

```bash
  python3 main.py
```


# USO

> **init(layers_size)**

>> inicializa a rede com as dimensões de cada camada (entrada, ocultas, saída)

>> ex.:  _init([2, 2, 1])_

> **save(filename)**

>> salva a solução incumbente (em memória) em um arquivo

>> ex.:  _save("exercício1.py")_

> **test(inputs, outputs, plot=False, tolerance=1e-4)**

>> testa um conjunto de entradas/saídas com a solução incumbente (em memória)

>> ex.:  _test([[1, 0], [0, 1], [1, 1], [0, 0]], [[1], [1], [0], [0]])_


> **train_batch(inputs, outputs, eta=0.55, maxit=1000, momentum=0.1, plot=False)**

>> faz o treinamento do modelo em modo batch

>> ex.: _train_batch([[1, 0], [0, 1], [1, 1], [0, 0]], [[1], [1], [0], [0]])_
