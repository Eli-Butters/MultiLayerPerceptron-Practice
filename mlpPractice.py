import pandas as pd
import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, numInputs, numHidden, numOutput, learning_rate):
        self.numI = numInputs
        self.numH = numHidden
        self.numO = numOutput

        self.weights_IH = Matrix(self.numH, self.numI)
        self.weights_HO = Matrix(self.numO, self.numH)
        self.weights_IH.randomize()
        self.weights_HO.randomize()

        self.bias_h = Matrix(self.numH, 1)
        self.bias_O = Matrix(self.numO, 1)
        self.bias_h.randomize()
        self.bias_O.randomize()

        self.learning_rate = learning_rate

    def feedforward(self, input_array):

        #Prepare Inputs
        inputs = Matrix.fromArray(input_array)

        #Hidden Layer
        hidden = self.weights_IH.dot(inputs)
        hidden.add(self.bias_h)
        hidden = hidden.map(sigmoid)

        #Output Layer
        output = self.weights_HO.dot(hidden)
        output.add(self.bias_O)
        output = output.map(sigmoid)

        return output.toArray()

    def train(self, inputs, targets):
        #Same as Feedforward()
        ##################################
        #Prepare Inputs
        inputs = Matrix.fromArray(inputs)

        #Hidden Layer
        hidden = self.weights_IH.dot(inputs)
        hidden.add(self.bias_h)
        hidden = hidden.map(sigmoid)

        #Output Layer
        outputs = self.weights_HO.dot(hidden)
        outputs.add(self.bias_O)
        outputs = outputs.map(sigmoid)

        #Convert to matrix object
        targets = Matrix.fromArray(targets)

        #####################################

        #Subtract
        output_error = Matrix.subtract(targets, outputs)

        #Dot Product of Weights and Errors
        HO_t = self.weights_HO.transpose()
        hidden_errors = HO_t.dot(output_error)

        #Find Gradient of HO
        gradientH = outputs.map(dsigmoid)
        gradientH.multiply(output_error)
        gradientH.multiply(self.learning_rate)

        #Find Delta of HO
        hiddenT = hidden.transpose()
        HOweights_D = gradientH.dot(hiddenT)

        #Change Bias
        self.bias_O.add(gradientH)

        #Change Weights
        self.weights_HO.add(HOweights_D)

        #Find Gradient of IH
        gradientI = hidden.map(dsigmoid)
        gradientI.multiply(hidden_errors)
        gradientI.multiply(self.learning_rate)

        #Find Delta of IH
        InputT = inputs.transpose()
        IHweights_D = gradientI.dot(InputT)

        #Change Bias
        self.bias_h.add(gradientI)

        #Change Weights
        self.weights_IH.add(IHweights_D)

        return

        
#CODE FOR MATRIX OPERATIONS IS REDUNDANT AND SLOW
#THIS IS JUST FOR MY FURTHER UNDERSTANDING

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = []

        #MATRIX INITIALIZATION

        for x in range(rows):
            self.matrix.append([])
            for i in range(cols):
                self.matrix[x].append(1)

    def fromArray(arr):
        m = Matrix(len(arr), 1)
        for i in range(len(arr)):
            m.matrix[i][0] = arr[i]
        return m

    def toArray(self):
        arr = []
        for x in range(len(self.matrix)):
            for i in range(len(self.matrix[x])):
                arr.append(self.matrix[x][i])
        return arr

    def multiply(self, n):
        if isinstance(n, Matrix):
            #Element-Wise Multiplication
            for x in range(len(self.matrix)):
                for i in range(len(self.matrix[x])):
                    self.matrix[x][i] *= n.matrix[x][i]
        else:
            #Scalar Multiplication
            for x in range(len(self.matrix)):
                for i in range(len(self.matrix[x])):
                    self.matrix[x][i] *= n

    def add(self, n):
        if isinstance(n, Matrix):
            #Element-Wise Addition
            for x in range(len(self.matrix)):
                for i in range(len(self.matrix[x])):
                    self.matrix[x][i] += n.matrix[x][i]
        else:
            #Scalar Addition
            for x in range(len(self.matrix)):
                for i in range(len(self.matrix[x])):
                    self.matrix[x][i] += n

    def subtract(one, two):
        if isinstance(one, Matrix):
            #Element-Wise Subtraction
            new = Matrix(one.rows, one.cols)
            for x in range(len(one.matrix)):
                for i in range(len(one.matrix[x])):
                    new.matrix[x][i] = one.matrix[x][i] - two.matrix[x][i]
        return new


    def dot(self, n):
        if isinstance(n, Matrix):
            #Dot Product
            if self.cols != n.rows:
                return TypeError
            result = Matrix(self.rows, n.cols)
            for x in range(len(result.matrix)):
                for i in range(len(result.matrix[x])):
                    #result matrix
                    sum = 0
                    for j in range(self.cols):
                        sum += self.matrix[x][j] * n.matrix[j][i]
                    result.matrix[x][i] = sum
            return result
        else:
            return TypeError

    def randomize(self):
        for i in range(self.rows):
            for x in range(self.cols):
                self.matrix[i][x] = (np.random.rand() * 2) - 1

    def transpose(self):
        #Transpose
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for x in range(self.cols):
                result.matrix[x][i] = self.matrix[i][x]
        return result

    def print(self):
        #Pandas DataFrames serves as a nice visual representation
        print(pd.DataFrame(self.matrix))

    def map(self, func):
        #Apply a function to every value
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for x in range(self.cols):
                result.matrix[i][x] = func(self.matrix[i][x])
        return result

########################################################
##########################MAIN##########################
########################################################

training_data = [
    [
        [0, 1],
        [1]
    ],
    [
        [1, 0],
        [1]
    ], 
    [
        [0, 0],
        [0]
    ],
    [
        [1, 1],
        [0]
    ]
]


nn = NeuralNetwork(2, 2, 1, 0.2)

for i in range(10000):
    for i in range(len(training_data)):
        y = math.floor((np.random.rand() * 4))
        nn.train(training_data[y][0], training_data[y][1])

sum_correct = 0
sum_wrong = 0

for i in range(1000):
    y = math.floor((np.random.rand() * 2))
    x = math.floor((np.random.rand() * 2))
    if (y == 0 and x == 0) or (y == 1 and x == 1):
        correct = 0
    else:
        correct = 1

    prediction = math.floor(nn.feedforward([y, x])[0] * 2)

    if (prediction == correct):
        sum_correct += 1
    else:
        print([y, x])
        print(prediction)
        print(correct)
        print()

        sum_wrong += 1

print(f'Sum Correct: {sum_correct}')
print(f'Sum Wrong: {sum_wrong}')

print(nn.feedforward([0, 0]))
print(nn.feedforward([1, 1]))
print(nn.feedforward([0, 1]))
print(nn.feedforward([1, 0]))