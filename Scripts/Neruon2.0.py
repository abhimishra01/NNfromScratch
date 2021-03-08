import numpy as np
inputs = [1.2, 3.6, -2.1, 3.2]
weights = [[-2, 0.6, -6.01, 1.2],
        [-2, 0.6, -6.01, 1.2],
        [-2, 0.6, -6.01, 1.2]]
biases = [-2,0.2,0.1]
def layer_output(inputs, weights, biases):
    layer_outputs = []
    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for n_input, n_weight in zip(inputs,neuron_weights):
            neuron_output += n_input*n_weight
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)
    return np.array(layer_outputs)

# simpler way of doing the same using dot product 
def layer_output_dot(inputs, weights, biases):
    layer_output = np.dot(weights, inputs) + biases
    return layer_output

layerOutput = layer_output(inputs, weights, biases)
layerOutput2 = layer_output_dot(inputs, weights, biases)

print(layerOutput)
print(layerOutput2)
# both are same 