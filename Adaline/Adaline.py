# coding=utf-8

import random
import numpy as np
import matplotlib.pyplot as plot

def calcStraight(weight, axis='x'):
    x = np.arange(-2, 3, 1)
    y = [-(weight[1] / weight[2]) * i -(weight[0] / weight[1]) for i in x]

    return y if axis == 'y' else x

class adalineAlgorithm(object):
    """adaline neuron"""
    def __init__(self, learningRate=0.05,
                 minError=0.000005, bias=-0.8649, maxIterations=10000,
                 weights=[-1, 0.3192, -0.8649]):

        self.learningRate = learningRate
        self.minError = minError
        self.bias = bias
        self.maxIterations = maxIterations

        self.weights = weights

    def calc_U(self, trainginData):
        output = 0
        for j in range(0, len(trainginData)):
                output += trainginData[j] * self.weights[j]
        return output

    def plotGraph(self, subplot, axis, trainingData, epoch):
        plot.subplot(subplot)
        plot.axis(axis)
        plot.title("Epoca " + str(epoch))
        plot.plot(calcStraight(self.weights, 'x'), calcStraight(self.weights, 'y'))
        for i in range(0, len(trainingData)):
            plot.plot(trainingData[i][1], trainingData[i][2], "o")

    def train(self, trainingData, expected):
        totalError = 0
        lastIter = 0
        # Plot first ephoc
        self.plotGraph(221, [-2, 2, -2, 2], trainingData, 1)

        self.plotGraph(222, [-2, 2, -2, 2], trainingData, 'n')
        for iters in range(1, self.maxIterations):
            plot.plot(calcStraight(self.weights, 'x'), calcStraight(self.weights, 'y'))
            lastIter = iters
            lastError = totalError
            totalError = 0
            print('interações ', iters)

            uv = []
            for i in range(0, len(trainingData)):
                plot.plot(trainingData[i][1], trainingData[i][2], "o")
                u = self.calc_U(trainingData[i])
                totalError += pow((expected[i] - u), 2)
                uv.append(u)
                # Alternative method to Adjust weights -- Less code, but the convergence is slower
                # self.weights = self.weights + ((self.learningRate * (expected[i] - u)) * trainingData[i])

            # Orthodox method to Adjust weights -- More code, but the convergence is faster
            deltaE = (expected[0] - uv[0]) * trainingData[0]
            for j in range(1, len(trainingData)):
                deltaE += (expected[j] - uv[j]) * trainingData[j]

            self.weights = self.weights + (self.learningRate * deltaE )
            # End Orthodox method

            # Stopping criterion -- the difference of this value to the value calculated in last
            # epoch should be smaller than `minError`
            totalError = totalError / len(trainingData)
            if abs(totalError-lastError) < self.minError:
                break

        self.plotGraph(223, [-2, 2, -2, 2], trainingData, lastIter)
        plot.show()


# Configuring and executing
bias = 0
trainingData = [ np.array([-1, 0, 0]),
                 np.array([-1, 0, 1]),
                 np.array([-1, 1, 0]),
                 np.array([-1, 1, 1]) ]
expected = np.array([0, 1, 1, 1])

weights = np.array([-1, 1, 1])
adaline = adalineAlgorithm(weights = weights, bias = bias, learningRate=0.2)
adaline.train(trainingData, expected)