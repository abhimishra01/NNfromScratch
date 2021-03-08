import numpy as np

def perceptron(inputs, weights):
    output = np.dot(inputs, weights)
    return output

inputs = [1.2, 3.6, -2.1, 3.2]
weights = [-2, 0.6, -6.01, 1.2]
bias = 0.4
# y = wx + b


output = perceptron(inputs, weights)
print(output)