#!/usr/bin/python

from random import random
from numpy import *


class MlpObject(object):
    def __init__(self, n_in, n_hid, n_out):
        self.n_Inputs = n_in + 1  # number of inputs
        self.n_Hidden = n_hid  # number of hidden units
        self.n_Outputs = n_out  # number of outputs

        self.dweights_1 = []
        self.dweights_2 = []

        self.active_low, self.active_hidden, self.active_out = [], [], []
        self.active_low = [1.0] * self.n_Inputs  # the lower layer
        self.active_hidden = [1.0] * self.n_Hidden  # the hidden layer
        self.active_out = [1.0] * self.n_Outputs  # the upper layer

        self.weights_1 = []
        self.weights_2 = []

        self.hidden = []
        self.outputs = []

        for i in range(self.n_Inputs):
            self.hidden.append([0.0] * self.n_Hidden)
            self.weights_1.append([0.0] * self.n_Hidden)

        for i in range(self.n_Hidden):
            self.outputs.append([0.0] * self.n_Outputs)
            self.weights_2.append([0.0] * self.n_Outputs)

    def randomise(self):
        for i in range(len(self.weights_1)):
            for j in range(len(self.weights_1[0])):
                self.weights_1[i][j] = random.random()

        for i in range(len(self.weights_2)):
            for j in range(len(self.weights_2[0])):
                self.weights_2[i][j] = random.random()

        self.dweights_1 = [0.0] * self.n_Hidden
        self.dweights_2 = [0.0] * self.n_Outputs

    def sigmoid(self, x, deriv=False):
        if deriv:
            return 1 - x ** 2
        return tanh(x)

    def forward(self, I):  # forward step
        for i in range(self.n_Inputs - 1):
            self.active_low[i] = I[i]

        for j in range(self.n_Hidden):
            sum = 0.0
            for i in range(self.n_Inputs):
                sum += (self.active_low[i] * self.weights_1[i][j])
            self.active_hidden[j] = self.sigmoid(sum)

        for k in range(self.n_Outputs):
            sum = 0.0
            for j in range(self.n_Hidden):
                sum += (self.active_hidden[j] * self.weights_2[j][k])
            self.active_out[k] = self.sigmoid(sum)
        return self.active_out

    def backwards(self, t):  # backward step
        for k in range(self.n_Outputs):
            error = t[k] - self.active_out[k]
            self.dweights_2[k] = error * self.sigmoid(self.active_out[k], deriv=True)
        for j in range(self.n_Hidden):
            error = 0.0
            for k in range(self.n_Outputs):
                error += self.dweights_2[k] * self.weights_2[j][k]
            self.dweights_1[j] = error * self.sigmoid(self.active_hidden[j], deriv=True)
        error = 0.0
        for k in range(len(t)):
            error = 0.5 * (t[k] - self.active_out[k]) ** 2
        return error

    def update_weights(self, N):
        for j in range(self.n_Hidden):
            for k in range(self.n_Outputs):
                change = self.dweights_2[k] * self.active_hidden[j]
                self.weights_2[j][k] += N * change + 0.1 * self.outputs[j][k]
                self.outputs[j][k] = change
        for i in range(self.n_Inputs):
            for j in range(self.n_Hidden):
                change = self.dweights_1[j] * self.active_low[i]
                self.weights_1[i][j] += N * change + 0.1 * self.hidden[i][j]
                self.hidden[i][j] = change
        self.dweights_1 = [0.0] * self.n_Hidden
        self.dweights_2 = [0.0] * self.n_Outputs

    def getOutputs(self):
        return self.active_out
