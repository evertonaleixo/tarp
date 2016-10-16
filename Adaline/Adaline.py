# coding=utf-8

import random
#import matplotlib.pyplot as plt


class adalineAlgorithm(object):
    """adaline neuron"""
    def __init__(self, numInputElements=4, numOutputElements=4, learningRate=0.099, minError=.13, bias=0.532, maxIterations=4000, outputElements=[0, 1, 1, 1], weights=[0.54232,0.5829]):
        self.numOutputElements = numInputElements
        self.numOutputElements = numOutputElements
        self.learningRate = learningRate
        self.minError = minError
        self.bias = bias
        self.maxIterations = maxIterations

        self.outputElements = outputElements
        self.weights = self.initWeights()

    def initWeights(self):
            weights = []
            for randomWeights in range(0, self.numOutputElements):
                weights.append(random.randint(1, 50))
            return weights

    def procces(self, trainginData, iters):
        output = 0
        for j in range(0, len(trainginData[iters])):
                output += trainginData[iters][j] * self.weights[j]
        return output

    def train(self, trainginData):
        for iters in range(1, self.maxIterations):
            print('interações')
            print (iters)
            output = 0
            for i in range(0, len(trainginData)):
                output = self.procces(trainginData, i)
                desiredOutput = output
                print (output)
                for data in range(0, len(self.weights)):
                    self.weights[data] = self.weights[data] + self.learningRate * (self.outputElements[i] - desiredOutput) * trainginData[i][data]

trainginData = [[1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]]
adaline = adalineAlgorithm()
adaline.train(trainginData)