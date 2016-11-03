# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
class ManipulandoGraficos:
    def __init__(self):
        self.vetor1 = []
        self.vetor2 = []

    #adiciona no gráfico com valores azuis
    def adicionarPontos(self, plt, vetor1, vetor2):
        plt.plot(vetor1, vetor2, 'bo')
        return plt

    #Essa função é apenas para dados de duas dimensões
    def adicionaPontosVetor(self, plt, vetor):
        if len(vetor) == 2:
            if vetor[0] == 0 and vetor[1] == 0:
                # apenas para mudar a cor para vermelho quando é 0 e 0
                plt.plot(vetor[0], vetor[1], 'ro')
            else:
                #apenas para mudar a cor para azul quando o eixo X e Y são diferentes de 0
                plt.plot(vetor[0],vetor[1], 'bo')
        else:
            print ("ERRO, função para dados de duas dimensões")
            return False
        return plt

    def adicionarReta(self, plt, vetor1, vetor2):
        plt.plot(vetor1, vetor2)
        return plt

    def criandoGraficoVetores(self, vetor1, vetor2):
        self.adicionarPontos(plt, vetor1, vetor2)
        self.adicionarReta(plt, vetor1, vetor2)
        plt.show()

    def pegandoOsMaioresValoresPorColuna(self, matriz):
        doubleMaioresValoresPorColuna = np.amax(matriz, axis=0)
        # isso é para transformar em um vetor normal, ou seja,x = np.array([[1,2]])
        # x = [[1,2]] agora fazendo x[0] eu pego apenas [1,2], então retorno apenas o [1,2]
        doubleMaioresValoresPorColuna = np.array(doubleMaioresValoresPorColuna[0])[0]
        return doubleMaioresValoresPorColuna

    def pegandoOsMenoresValoresPorColuna(self, matriz):
        doubleMenoresValoresPorColuna = np.amin(matriz, axis=0)
        # isso é para transformar em um vetor normal, ou seja,x = np.array([[1,2]])
        # x = [[1,2]] agora fazendo x[0] eu pego apenas [1,2], então retorno apenas o [1,2]
        doubleMenoresValoresPorColuna = np.array(doubleMenoresValoresPorColuna[0])[0]
        return doubleMenoresValoresPorColuna

    def configurandoEixos(self, plt, matriz):
        #Depois posso tirar essa linha
        matriz = np.matrix(matriz)
        doubleMaioresValoresPorColuna = self.pegandoOsMaioresValoresPorColuna(matriz)
        doubleMenoresValoresPorColuna = self.pegandoOsMenoresValoresPorColuna(matriz)

        #Defino os intervalos, o -1 e o +1 é para criar um espaço entre os eixos menores e maiores
        plt.axis((doubleMenoresValoresPorColuna[0]-1, doubleMaioresValoresPorColuna[0] + 1, doubleMenoresValoresPorColuna[1] - 1,
doubleMaioresValoresPorColuna[1] + 1))
        return plt

    def adicionandoTodosOsPontosDaMatriz(self, plt, matriz):
        matriz = np.matrix(matriz)
        # função que pega o tamanho da matriz e retorna linha e coluna
        intQtdLinha, intQtdColuna = np.shape(matriz)
        for i in range(intQtdLinha):
            vetorUmaLinhaMatriz = np.array(matriz[i][0])
            # faço isso para retirar vetor dentro de vetor
            vetorUmaLinhaMatriz = np.array(vetorUmaLinhaMatriz[0])
            self.adicionaPontosVetor(plt, vetorUmaLinhaMatriz)
        return plt

    def configurandoEixos(self, plt, matriz):
        #Depois posso tirar essa linha
        matriz = np.matrix(matriz)
        doubleMaioresValoresPorColuna = self.pegandoOsMaioresValoresPorColuna(matriz)
        doubleMenoresValoresPorColuna = self.pegandoOsMenoresValoresPorColuna(matriz)

        #Defino os intervalos, o -1 e o +1 é para criar um espaço entre os eixos menores e maiores
        plt.axis((doubleMenoresValoresPorColuna[0]-1, doubleMaioresValoresPorColuna[0] + 1, doubleMenoresValoresPorColuna[1] - 1,
    doubleMaioresValoresPorColuna[1] + 1))
        return plt

    def configurandoEixosParaDoisVetores(self, plt, vetorX, vetorY):
        vetorX = np.array(vetorX)
        vetorY = np.array(vetorY)
        menorValorX = np.min(vetorX)
        maiorValorX = np.max(vetorX)
        menorValorY = np.min(vetorY)
        maiorValorY = np.max(vetorY)
        plt.axis((menorValorX - 0.1, maiorValorX + 0.1, menorValorY - 0.1, maiorValorY + 0.1))
        return plt

    def criandoGraficoComDuasRetas(self, vetorX, vetorY, strNomeEixoX, strNomeEixoY):
        print ("deu certo!")
        self.configurandoEixosParaDoisVetores(plt, vetorX, vetorY)
        plt.ylabel(strNomeEixoY)
        plt.xlabel(strNomeEixoX)
        plt.plot(vetorX, vetorY, "o-")
        plt.show()

    def criandoGraficoMatriz(self, matriz, vetorPesos, bias):
        # Configuro os eixos para melhorar a visão
        #self.configurandoEixos(plt, matriz)
        # Adiciono os pontos no meu gráfico
        print ("Adicionando os pontos..."),
        self.adicionandoTodosOsPontosDaMatriz(plt, matriz)
        print ("Ok!")
        # Faço isso para ser o meu vetor de entrada no plano cartesiano
        #por ex. X | F(X) = X*2
        # -2 | 4
        # -1 | 1
        # 0 | 0
        # 1 | 1
        # sendo que o meu F(X) será o vetor Y
        vetorX = np.array([-2, -1, 0, 1, 2])

        # Essa vai ser minha saída
        vetorY = self.criandoValoresParaOEixoY(vetorX, vetorPesos, bias)

        #ploto o gráfico
        plt.plot(vetorX, vetorY, '-rx')

        print ("mostrando gráfico... "),
        # quando eu coloco o show.. automáticamente descarrega o buffer do gráfico
        plt.show()
        print ("ok!")

    def criandoValoresParaOEixoY(self, vetorX, vetorPesos, bias):
        tamX = len(vetorX)
        vetorY = np.array([0.0 for x in range(tamX)])
        for i in range(tamX):
            vetorY[i] = self.criandoValorParaY(vetorX[i], vetorPesos, bias)
        print ("vetor para Y: ", vetorY)
        return vetorY

    def criandoValorParaY(self, doubleX, vetorPesos, bias):
        return np.double(-(bias/vetorPesos[1]) - vetorPesos[0] * doubleX /vetorPesos[1])
