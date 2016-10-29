# coding=utf-8

import random
#import matplotlib.pyplot as plt

class multilayerPerceptronAlgorithm(object):
    def __init__(self, learningRate=0.099, minError=.13, bias=0.532, maxIterations=4000, initialInput=[[]], expectedOutput=[[]], layersDescription=[ [3,3], [2,2,2], [3] ]):
        self.layersDescription = layersDescription
        self.initialInput = initialInput
        self.minError = minError
        self.bias = bias
        self.maxIterations = maxIterations
        self.expectedOutput = expectedOutput
        self.learningRate = learningRate

        self.weights = self.initWeights()

    def initWeights(self):
        weights = [[[]]]

        for layer_num in range(0, self.layersDescription.__len__()):
            for neurone_num in range(0, self.layersDescription[layer_num].__len__()):
                for dendrites_num in range(0, self.layersDescription[layer_num][neurone_num]):
                    weights[layer_num][neurone_num].append(random.randint(1, 50))
        return weights

    def printWeights(self):
        for layer_num in range(0, self.layersDescription.__len__()):
            for neurone_num in range(0, self.layersDescription[layer_num].__len__()):
                for dendrites_num in range(0, self.layersDescription[layer_num][neurone_num]):
                    print self.weights[layer_num][neurone_num][dendrites_num], " "
                print ""
            print "\n"



pmc = multilayerPerceptronAlgorithm()

pmc.printWeights()