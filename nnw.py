import numpy as np

class Neuron:
    def __init__(self, input_size):
        # Initialize weights and bias
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def forward(self, inputs):
        # Compute the weighted sum of inputs and bias
        self.output = np.dot(inputs, self.weights) + self.bias
        # Apply the activation function (e.g., sigmoid)
        return self.sigmoid(self.output)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class Layer:
    def __init__(self, num_neurons, input_size):
        # Initialize a layer with given number of neurons
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def forward(self, inputs):
        # Forward the inputs through each neuron in the layer
        self.outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return self.outputs


class NeuralNetwork:
    def __init__(self, layer_sizes):
        # Initialize the network with layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i+1], layer_sizes[i]))

    def forward(self, inputs):
        # Forward the inputs through each layer in the network
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


# Example usage:
# Define a network with 3 input neurons, one hidden layer with 5 neurons, and 2 output neurons
nn = NeuralNetwork([3, 5, 2])

# Forward pass with example input
inputs = np.array([1.0, 2.0, 3.0])
output = nn.forward(inputs)
print(output)
