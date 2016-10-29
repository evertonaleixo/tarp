# coding=utf-8
# perceptron2.py
# aplicativo para analise de portas OR

import numpy as np
import matplotlib.pyplot as plot

# falso = 0, verdadeiro = 1

# [0,0] | resposta = 0
# [0,1] | resposta = 1
# [1,0] | resposta = 1
# [1,1] | resposta = 1

def calcStraight(weight, axis='x'):
    x = np.arange(-2, 3, 1)
    y = [-(weight[1] / weight[2]) * i -(weight[0] / weight[1]) for i in x]

    return y if axis == 'y' else x

# numero maximo de interacoes
max_int = 20

# threshold (limiar)
threshold = 0

# peso 0 bias
w_0 = -0.8649

# entrada 0 bias
x_0 = 1

# entradas treino
x = [[x_0,0,0],
     [x_0,0,1],
     [x_0,1,0],
     [x_0,1,1]]

# respostas desejadas
d = [0,1,1,1]

# quantos itens tem o vetor x (4)
tamanho_x = len(x)

# quantos itens estão em cada posicao do vetor x
qtde_itens_x = len(x[0])

# pesos (sinapses)
w = [w_0, 0.3192, 0.3129]

# quantos itens tem o vetor w (3)
tamanho_w = len(w)


# taxa de aprendizado (n)
taxa_aprendizado = 0.5

#saida do peceptron
y = 0

# resposta = acerto ou falha
resposta = ""

# soma (funcao agregadora
u = 0

#erros encontrados
e = 0

print("Treinando")

# inicio do algoritmo
for k in range(1,max_int):
    print("INTERACAO "+str(k)+"-------------------------")
    plot.subplot(220 + k)
    plot.axis([-2, 2, -2, 2])

    acertos = 0
    e = 0
    plot.plot(calcStraight(w, 'x'), calcStraight(w, 'y'))
    plot.title("Epoca" + str(k))
    for t in range(0,tamanho_x):
        plot.plot(x[t][1], x[t][2], "o")
        u = 0

        # para calcular a saida do perceptron, cada entrada de x eh multiplicada
        # pelo seu peso w correspondente
        for j in range(0,qtde_itens_x):
            u += x[t][j] * w[j]

        # funcao de saida
        if u > 0:
            y = 1
        else:
            y = 0

        # atualiza os pesos caso a saida nao corresponda ao valor esperado
        if y == d[t]:
            resposta = "acerto"
            acertos += 1
            e = 0
        else:
            resposta = "erro"
            # calculando o erro
            e = d[t] - y
            # atualizando os pesos
            for j in range(0, tamanho_w):
                w[j] = w[j] + (taxa_aprendizado * e * x[t][j])

        print(resposta + " >>> u = " + str(u) + ", y = " + str(y) + ", e = " + str(e))

    if acertos == tamanho_x:
        print("\nFuncionalidade aprendida com " + str(k) + " interacoes")
        print("\nPesos encontrados =============== ")
        for j in range(0, tamanho_w):
            print(w[j])
        break;
    print("")

plot.show();
print("Finalizado")