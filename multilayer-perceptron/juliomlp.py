import numpy as np

#===============================================================================#


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def limite_bin(x):
    if x <= 0.0:
        return 0.0
    return 1.0


def df_sigmoid(x):
    return np.exp(x)/((1.0 + np.exp(x))**2)

#===============================================================================#


def mlp_training(entrada, saida, f_act_saida=sigmoid, f_act_ocult=sigmoid, df_act=df_sigmoid, num_node_ocult=4, delta=0.5, eps_global=1e-4, eps_local=1e-2):  # jjj 1e-1 1e-1
    n_inst = len(entrada)
    n_entrada = len(entrada[0])
    n_saida = len(saida[0])

    if (n_inst != len(saida)):
        print("\n\n[ERRO]: Número de saidas diferente do número de entradas\n\n")
        exit(1)

    entrada_p = np.array(entrada)
    saida_p = np.array(saida)

    pesos_ocult = np.random.random((n_entrada, num_node_ocult))
    pesos_saida = np.random.random((num_node_ocult, n_saida))
    bias_ocult = np.random.random(num_node_ocult)
    bias_saida = np.random.random(n_saida)

    pesos_ocult_delta = np.zeros((n_entrada, num_node_ocult))
    pesos_saida_delta = np.zeros((num_node_ocult, n_saida))
    bias_ocult_delta = np.zeros(num_node_ocult)
    bias_saida_delta = np.zeros(n_saida)

    sigma_ocult = np.zeros(num_node_ocult)
    sigma_saida = np.zeros(n_saida)
    erro = np.ones(n_saida)

    input_acum_ocult = np.zeros(num_node_ocult)
    output_ocult = np.zeros(num_node_ocult)

    input_acum_saida = np.zeros(n_saida)
    output_saida = np.zeros(n_saida)
    jcount = 0  # jjj
    while(np.sum(np.abs(erro)) > eps_global):
        jcount += 1  # jjj
        # jjj
        print(f"While {jcount} | Erro = {np.sum(np.abs(erro))}\r", end="")
        for i_index in range(n_inst):
            # forward
            for node_ocult in range(num_node_ocult):
                input_acum_ocult[node_ocult] = np.sum(
                    entrada_p[i_index]*pesos_ocult[:, node_ocult])+bias_ocult[node_ocult]
                output_ocult[node_ocult] = f_act_ocult(
                    input_acum_ocult[node_ocult])

            for node_saida in range(n_saida):
                input_acum_saida[node_saida] = np.sum(
                    output_ocult*pesos_saida[:, node_saida])+bias_saida[node_saida]
                output_saida[node_saida] = f_act_saida(
                    input_acum_saida[node_saida])

            erro = saida_p[i_index] - output_saida

            if (np.sum(np.abs(erro)) <= eps_local):
                continue
        # backward

            for node_saida in range(n_saida):
                sigma_saida[node_saida] = erro[node_saida] * \
                    df_act(input_acum_saida[node_saida])
                pesos_saida_delta[:, node_saida] = delta * \
                    sigma_saida[node_saida]*output_ocult
                bias_saida_delta[node_saida] = delta*sigma_saida[node_saida]

            for node_ocult in range(num_node_ocult):
                sigma_ocult[node_ocult] = np.sum(
                    sigma_saida*pesos_saida[node_ocult, :])*df_act(input_acum_ocult[node_ocult])
                pesos_ocult_delta[:, node_ocult] = delta * \
                    sigma_ocult[node_ocult]*entrada_p[i_index]
                bias_ocult_delta[node_ocult] = delta*sigma_ocult[node_ocult]

            pesos_ocult = pesos_ocult + pesos_ocult_delta
            bias_ocult = bias_ocult + bias_ocult_delta
            pesos_saida = pesos_saida + pesos_saida_delta
            bias_saida = bias_saida + bias_saida_delta

            break

    def mlp(x):
        output_ocult_mlp = np.zeros(num_node_ocult)
        output_mlp = np.zeros(n_saida)
        for node_ocult in range(num_node_ocult):
            output_ocult_mlp[node_ocult] = f_act_ocult(
                np.sum(x*pesos_ocult[:, node_ocult])+bias_ocult[node_ocult])

        for node_saida in range(n_saida):
            output_mlp[node_saida] = f_act_saida(
                np.sum(output_ocult_mlp*pesos_saida[:, node_saida])+bias_saida[node_saida])

        return output_mlp
    return mlp

#===============================================================================#


a = [[0, 0], [0, 1], [1, 0], [1, 1]]
b = [[0], [1], [1], [0]]

mlp = mlp_training(a, b,)
print(round(mlp([0, 0])[0]))
print(round(mlp([0, 1])[0]))
print(round(mlp([1, 0])[0]))
print(round(mlp([1, 1])[0]))
