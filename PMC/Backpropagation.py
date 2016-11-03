# coding=utf-8
import numpy as np

from PMC import ManipulandoGraficos

class Backpropagation:
    def __init__(self, doubleVetorEntradas, doubleVetorDePesosPrimeiraCamada, doubleVetorDePesosSegundaCamada, doubleVetorSaida,
    doubleBiasPrimeiraCamada, doubleBiasSegundaCamada, doubleTaxaDeAprendizagem, doublePrecisaoRequerida, boolMostrarGraficos,
    intNumerosDeIteracao):
        self.doubleVetorEntradas = np.array(doubleVetorEntradas)
        self.doubleVetorDePesosPrimeiraCamada = np.array(doubleVetorDePesosPrimeiraCamada)
        self.doubleVetorPesosW1eW2 = np.array([doubleVetorDePesosPrimeiraCamada[0], doubleVetorDePesosPrimeiraCamada[1]])
        self.doubleVetorPesosW3eW4 = np.array([doubleVetorDePesosPrimeiraCamada[2], doubleVetorDePesosPrimeiraCamada[3]])
        self.doubleVetorDePesosSegundaCamada = np.array(doubleVetorDePesosSegundaCamada)
        self.doubleVetorSaida = np.array(doubleVetorSaida)
        self.doubleBiasPrimeiraCamada = np.double(doubleBiasPrimeiraCamada)
        self.doubleBiasSegundaCamada = np.double(doubleBiasSegundaCamada)
        self.doubleTaxaDeAprendizagem = np.double(doubleTaxaDeAprendizagem)
        self.doublePrecisaoRequerida = np.double(doublePrecisaoRequerida)
        self.boolMostrarGraficos = boolMostrarGraficos
        self.intNumerosDeIteracao = intNumerosDeIteracao
    #Faz o sum(entrada*peso) + bias
    def calculaNet(self,vetorEntrada, vetorPeso, bias):
        return np.sum(vetorEntrada * vetorPeso) + bias
        #faz 1/(1+e^{-NET})
    def calculaFuncaoSinoide_Gh(self, doubleNET):
        return 1/(1 + np.exp(-doubleNET))

    def neuronio(self, doubleVetorEntrada, doubleVetorPesosCamada, doubleBiasCamada):
        doubleVetorPesosW1eW2 = np.array([doubleVetorPesosCamada[0], doubleVetorPesosCamada[1]])
        doubleVetorPesosW3eW4 = np.array([doubleVetorPesosCamada[2], doubleVetorPesosCamada[3]])

        ############## CALCULANDO OS NET's da CAMADA ##############
        doubleNETh1 = self.calculaNet(doubleVetorEntrada, doubleVetorPesosW1eW2, doubleBiasCamada)
        doubleNETh2 = self.calculaNet(doubleVetorEntrada, doubleVetorPesosW3eW4, doubleBiasCamada)

        ############### CALCULANDO OS Gh1 e Gh2 da CAMADA ###############
        doubleGh1 = np.double(self.calculaFuncaoSinoide_Gh(doubleNETh1))
        doubleGh2 = np.double(self.calculaFuncaoSinoide_Gh(doubleNETh2))
        return doubleNETh1, doubleNETh2, doubleGh1, doubleGh2

    def calculandoErroTotal(self, doubleG1, doubleG2, doubleVetorSaida):
        #Calculando Erro 1
        print ("doubleVetorSaida[0]", doubleVetorSaida[0])
        doubleErro1 = np.double(1.0/2.0 * ((doubleG1 - doubleVetorSaida[0]) * (doubleG1 - doubleVetorSaida[0])))
        doubleErro2 = np.double(1.0/2.0 * ((doubleG2 - doubleVetorSaida[1]) * (doubleG2 - doubleVetorSaida[1])))
        print ("doubleErro1:", doubleErro1, " doubleErro2: ", doubleErro2)
        return doubleErro1 + doubleErro2

    def correcaoDoErroWSegundaCamada(self, doubleValorDeSaida, doubleG_primeiraCamada, doubleG_segundaCamada):
        x = (-(doubleValorDeSaida - doubleG_segundaCamada[2]) * doubleG_segundaCamada[2] * (1 - doubleG_segundaCamada[3]) * doubleG_primeiraCamada)
        return x
    def calcularDerivadaErro_NET(self, doubleValorG_segundaCamada, doubleValorSaida):
        return (doubleValorG_segundaCamada - doubleValorSaida) * (doubleValorG_segundaCamada * (1 - doubleValorG_segundaCamada))

    def calculaB_derivadaGh_NETh(self, doubleValorG):
        return doubleValorG * (1 - doubleValorG)

    def calculaA_ErroTotal_g(self, doubleVetorPesos, doubleVetorG_segundaCamada, doubleVetorSaida):
    #Calcula A1.1, A1.2, multiplica pelos pesos e faz a soma para o erro total""" Este for equivale a isso, veja o PDF da disciplina
        # Calcula A1.1
        doubleA1_1w1 = self.calcularDerivadaErro_NET(doubleVetorG_segundaCamada[0], doubleVetorSaida[0])
        print ("doubleA1_1w1: ", doubleA1_1w1)
        #Calcula A2.1
        doubleA2_1_w1 = self.calcularDerivadaErro_NET(doubleVetorG_segundaCamada[1], doubleVetorSaida[1])
        print ("doubleA2_1_w1: ", doubleA2_1_w1)
         #Calcula A w1
        doubleA_w1 = doubleA1_1w1 * self.doubleVetorDePesosSegundaCamada[0] + doubleA2_1_w1 * self.doubleVetorDePesosSegundaCamada[2]
        print ("doubleAw1: ", doubleA_w1)
        intTam = len(doubleVetorG_segundaCamada)
        doubleErroTotal = np.double(0.0)
        for i in range(intTam):
            doubleErroTotal += self.calcularDerivadaErro_NET(doubleVetorG_segundaCamada[i], doubleVetorSaida[i]) * \
                               doubleVetorPesos[i]
        return doubleErroTotal

    def correcaoDoErroWPrimeiraCamada(self, doubleVetorEntradas, doubleVetorPesosSegundaCamada, doubleVetorSaida,doubleVetorG_primeiraCamada, doubleVetorG_segundaCamada):
        #Calcula A w1 e w2
        doubleA_w1_e_w2 = self.calculaA_ErroTotal_g(np.array([doubleVetorPesosSegundaCamada[0], doubleVetorPesosSegundaCamada[2]]),
        doubleVetorG_segundaCamada, doubleVetorSaida)
        print ("doubleA_w1_e_w2: ", doubleA_w1_e_w2)
        #Calcula B w1 e w2
        doubleB_w1_e_w2 = self.calculaB_derivadaGh_NETh(doubleVetorG_primeiraCamada[0])
        print ("doubleBw1: ", doubleB_w1_e_w2)
        #Calcula C_w1
        doubleC_w1 = doubleVetorEntradas[0]
        #Calcula ErroTotal_w1
        erroTotal_w1 = doubleA_w1_e_w2 * doubleB_w1_e_w2 * doubleC_w1
        print ("ErroTotal_w1: ", erroTotal_w1)
        # Calcula C_w2
        doubleC_w2 = doubleVetorEntradas[1]
        # Calcula ErroTotal_w2
        erroTotal_w2 = doubleA_w1_e_w2 * doubleB_w1_e_w2 * doubleC_w2
        print ("ErroTotal_w2: ", erroTotal_w2)
        doubleA_w3_e_w4 = self.calculaA_ErroTotal_g(np.array([doubleVetorPesosSegundaCamada[1], doubleVetorPesosSegundaCamada[3]]),
        doubleVetorG_segundaCamada, doubleVetorSaida)
        print ("doubleA_w3_e_w4: ", doubleA_w3_e_w4)
        #Calcula B w3 e w4
        doubleB_w3_e_w4 = self.calculaB_derivadaGh_NETh(doubleVetorG_primeiraCamada[1])
        print ("doubleB_w3_e_w4: ", doubleB_w3_e_w4)
        # Calcula C_w3
        doubleC_w3 = doubleVetorEntradas[0]
        # Calcula C_w4
        doubleC_w4 = doubleVetorEntradas[1]
        # Calcula ErroTotal_w3
        erroTotal_w3 = doubleA_w3_e_w4 * doubleB_w3_e_w4 * doubleC_w3
        print ("ErroTotal_w3: ", erroTotal_w3)
        # Calcula ErroTotal_w4
        erroTotal_w4 = doubleA_w3_e_w4 * doubleB_w3_e_w4 * doubleC_w4
        print ("ErroTotal_w4: ", erroTotal_w4)
        return erroTotal_w1, erroTotal_w2, erroTotal_w3, erroTotal_w4

    def atualizandoPesosDasCamadas(self, doubleVetorPesosOriginal, vetorErroTotalDosPesos, doubleTaxaDeAprendizagem):
        intTamVetor = len(doubleVetorPesosOriginal)
        for i in range(intTamVetor):
            doubleVetorPesosOriginal = doubleVetorPesosOriginal - (doubleTaxaDeAprendizagem * vetorErroTotalDosPesos[i])
        return doubleVetorPesosOriginal

    def execucaoAlgoritmo(self):
        print ("Execução algoritmo...")
        intEpocas = 1;
        boolContinueRodando = True
        vetorDeErros = []
        vetorDeEpocas = []
        while boolContinueRodando == True:
            print ("======================= Epoca "), intEpocas, " ======================"

            ########################### Neurônio da 1ª camada ###########################
            print ("1ª camada...")
            doubleNETh1, doubleNETh2, doubleGh1, doubleGh2 = self.neuronio(self.doubleVetorEntradas,self.doubleVetorDePesosPrimeiraCamada, self.doubleBiasPrimeiraCamada)
            print ("doubleNETh1: ", doubleNETh1, " doubleNETh2: ")

            ########################### Neurônio da 2ª camada #######################
            print ("2ª camada...")

            #Transformo o Gh1 e Gh2 nas minhas novas entradas
            doubleVetorG_primeiraCamada_Gh1_Gh2 = np.array([doubleGh1, doubleGh2])

            doubleNETo1 = self.neuronio(doubleVetorG_primeiraCamada_Gh1_Gh2, self.doubleVetorDePesosSegundaCamada, self.doubleBiasSegundaCamada)

            doubleNETo2 = self.neuronio(doubleVetorG_primeiraCamada_Gh1_Gh2,
                                                                   self.doubleVetorDePesosSegundaCamada,
                                                                   self.doubleBiasSegundaCamada)
            doubleGo1 = self.neuronio(doubleVetorG_primeiraCamada_Gh1_Gh2,
                                                               self.doubleVetorDePesosSegundaCamada,
                                                               self.doubleBiasSegundaCamada)
            doubleGo2 = self.neuronio(doubleVetorG_primeiraCamada_Gh1_Gh2,self.doubleVetorDePesosSegundaCamada,self.doubleBiasSegundaCamada)
            print ('doubleNETo1:')
           # print  doubleNETo1
            print (" doubleNETo2: ")
           # print doubleNETo2,
            print  (" doubleGo2: ")
           # print doubleGo2
            doubleVetorG_segundaCamada_Go1_Go2 = np.array([doubleGo1, doubleGo2])


            ### # Calculando o erro Total
            #ARRUMAR... colocar o vetor de G da segunda camada
            doubleErroTotal = self.calculandoErroTotal(doubleGo1, doubleGo2, self.doubleVetorSaida)
            print ("doubleErroTotal: ", doubleErroTotal)
            ## Correção do erro da segunda camada
            #ARRUMAR... colocar isso tudo em uma função, trocar o doubleGh1 e doubleGo1 por vetores e variar eles no laço
            doubleDerivadaErroTotal_w5 = self.correcaoDoErroWSegundaCamada(self.doubleVetorSaida[0], doubleGh1, doubleGo1)
            print ("doubleDerivadaErroTotal_w5: ", doubleDerivadaErroTotal_w5)
            doubleDerivadaErroTotal_w6 = self.correcaoDoErroWSegundaCamada(self.doubleVetorSaida[0], doubleGh2, doubleGo1)
            print ("doubleDerivadaErroTotal_w6: ", doubleDerivadaErroTotal_w6)
            doubleDerivadaErroTotal_w7 = self.correcaoDoErroWSegundaCamada(self.doubleVetorSaida[1], doubleGh1, doubleGo2)
            print ("doubleDerivadaErroTotal_w7: ", doubleDerivadaErroTotal_w7)
            doubleDerivadaErroTotal_w8 = self.correcaoDoErroWSegundaCamada(self.doubleVetorSaida[1], doubleGh2, doubleGo2)
            print ("doubleDerivadaErroTotal_w8: ", doubleDerivadaErroTotal_w8)


            ######### Correção do erro da primeira camada ##########
            doubleErroTotal_w1, doubleErroTotal_w2, doubleErroTotal_w3, doubleErroTotal_w4 = self.correcaoDoErroWPrimeiraCamada(self.doubleVetorEntradas, self.doubleVetorDePesosSegundaCamada, self.doubleVetorSaida, doubleVetorG_primeiraCamada_Gh1_Gh2, doubleVetorG_segundaCamada_Go1_Go2)
            print ("doubleErroTotal_w1: ", doubleErroTotal_w1)
            print (" doubleErroTotal_w2: ", doubleErroTotal_w2)
            print (" doubleErroTotal_w3: ",doubleErroTotal_w3)
            print (" doubleErroTotal_w4: ", doubleErroTotal_w4)


            ############# Atualização dos pesos ###############
            doubleVetorErroTotalPrimeiraCamada = np.array([doubleErroTotal_w1, doubleErroTotal_w2, doubleErroTotal_w3,doubleErroTotal_w4])
            doubleVetorErroTotalSegundaCamada = np.array([doubleDerivadaErroTotal_w5, doubleDerivadaErroTotal_w6,doubleDerivadaErroTotal_w7, doubleDerivadaErroTotal_w8])
            print ("doubleVetorErroTotalPrimeiraCamada: ", doubleVetorErroTotalPrimeiraCamada)
            print ("doubleVetorErroTotalSegundaCamada: ", doubleVetorErroTotalSegundaCamada)


            ## Atualizando primeira camada
            self.doubleVetorDePesosPrimeiraCamada = self.atualizandoPesosDasCamadas(self.doubleVetorDePesosPrimeiraCamada,doubleVetorErroTotalPrimeiraCamada, self.doubleTaxaDeAprendizagem)
            print ("doubleVetorDePesosPrimeiraCamada t+1: ", self.doubleVetorDePesosPrimeiraCamada)

            ## Atualizando segunda camada
            self.doubleVetorDePesosSegundaCamada = self.atualizandoPesosDasCamadas(self.doubleVetorDePesosSegundaCamada,doubleVetorErroTotalSegundaCamada, self.doubleTaxaDeAprendizagem)
            print ("doubleVetorDePesosSegundaCamada t+1: ", self.doubleVetorDePesosSegundaCamada)
            print ("doubleErroTotal: ", doubleErroTotal)

            ## Informações para criar o gráfico ##
            vetorDeErros.append(doubleErroTotal)
            vetorDeEpocas.append(intEpocas)
            if max(doubleErroTotal) < self.doublePrecisaoRequerida or self.intNumerosDeIteracao <= intEpocas:
                boolContinueRodando = False
            intEpocas = intEpocas + 1

        if self.boolMostrarGraficos == True:
            m = ManipulandoGraficos()
            m.criandoGraficoComDuasRetas(vetorDeEpocas, vetorDeErros, "Epocas", "Erro")

