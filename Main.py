#!/usr/bin/python

from MlpObject import MlpObject

XOR = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]]


class Main:
    def __init__(self):
        n_in = 2
        n_hid = 2
        n_out = 1
        self.NN = MlpObject(n_in, n_hid, n_out)
        self.NN.randomise()
        self.maxEpochs = 10000
        self.learningRate = 0.5
        self.sin_vector = [] * 2

    def evaluate(self):
        for e in range(self.maxEpochs):
            error = 0.0
            for arr in XOR:
                self.NN.forward(arr[0])
                error += self.NN.backwards(arr[1])
                if e % 10 == 0:
                    self.NN.update_weights(self.learningRate)
            if e % 1000 == 0:
                print "Error at epoch " + str(e) + " is " + str(error) + "\n"

    def test(self):
        for arr in XOR:
            self.NN.forward(arr[0])
            print 'Inputs:', arr[0], '\tTarget', arr[1], '\tPrediction', self.NN.getOutputs(), '\n'


run = Main()
run.evaluate()
run.test()
