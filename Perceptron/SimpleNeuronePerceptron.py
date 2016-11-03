# coding=utf-8
import numpy as np
import matplotlib.pyplot as plot


def degrau(u):
    if u < 0:
        return 0
    else:
        return 1


"""
Calcula a reta

Para cada ponto desejado no gráfico (i) é calculado onde a reta passaria

y = ax+b
onde b = -peso[0]/peso[1]
onde ax = bias/peso[1]

Poderia ter feito o calculo apenas duas vezes, 1 no ponto inicial desejado e outra no ponto final.
"""
def limiar2D(pesos, bias, axis):
    x = np.arange(-2, 3, 1) #vai de -2 a 2 no eixo x e para cada valor (-2,-1,,0,1 e 2) é calculado o seu valor em y
    y = [-(pesos[0] / pesos[1]) * i - (bias / pesos[1]) for i in x]
    return y if axis == 'y' else x


"""
Calcula o potencial de ativação da entrada
Se o potencial for < 0, retorna 0, caso contrário retorna 1
"""
def perceptron(entradas, pesos, bias, limiar):
    if len(entradas) != len(pesos):
        return False

    potencial = bias - limiar + np.sum(entradas * pesos)
    return degrau(potencial)

"""
Atualiza o  vetor de peso
"""
def calcularPesos(entradas, pesos, erro, taxa):
    if len(entradas) != len(pesos):
        return False

    return pesos + taxa * erro * entradas


def treinar(entradas, desejados, erroAceitavel, maxTentativas, pesosIniciais, bias, taxa):
    pesos = pesosIniciais
    print("Pesos iniciais:", pesos)

    plot.figure(1)

    erroTotal = 1
    tentativa = 1
    while erroTotal > erroAceitavel and tentativa <= maxTentativas:
        erroTotal = 0
        print("\nÉpoca ", tentativa, ":")
        plot.subplot(220 + tentativa)
        plot.axis([-2, 2, -2, 2])
        plot.plot(limiar2D(pesos, bias, 'x'), limiar2D(pesos, bias, 'y'))
        for i, entrada in enumerate(entradas):
            plot.plot(entrada[0], entrada[1], marker='o')
            plot.title("Época " + str(tentativa))
            obtido = perceptron(entrada, pesos, bias, 0)
            print("Entradas:", entrada, "| Obtido:", obtido, "| Desejado:", desejados[i])
            erro = desejados[i] - obtido
            erroTotal += abs(erro)
            if (abs(erro) > erroAceitavel):
                pesos = calcularPesos(entrada, pesos, erro, taxa)
                print("Alterando pesos para:", pesos)
        tentativa += 1
    print("\nPesos finais:", pesos)
    plot.show()


entradas = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
desejados = np.array([0, 1, 1, 1])
treinar(entradas, desejados, 0.00001, 10, [0.4590, 0.2110], -0.5130, 0.5)