from numpy import exp, random, dot


def sigmoid(sum):
    return 1 / (1 + exp(-sum))


def w_derivative(nn_output, true_output, data_input):
    return sum(2 * (true_output - nn_output) * (nn_output * (1 - nn_output)) * data_input)/len(data_input)


def b_derivative(nn_output, true_output):
    return sum(2 * (true_output - nn_output) * (nn_output * (1 - nn_output)))


def cost(nn_output, true_output):
    return sum((true_output - nn_output) * (true_output - nn_output))/len(data_input)


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


class Neuron():
    def __init__(self, inputs, neurons):
        self.weights = 2 * random.random((inputs, neurons)) - 1
        self.biases = 2 * random.random((1, neurons)) - 1

    def think(self, data):
        return sigmoid(dot(data, self.weights) + self.biases)

    def learn(self, data, data_output, times):
        for x in range(times):
            result = self.think(data)
            print(mean(result-data_output))
            w_slope = w_derivative(result, data_output, data)
            b_slope = b_derivative(result, data_output)
            adj = sum(w_slope * abs(self.weights))
            for i in range(len(adj)):
                self.weights[i] += adj[i]
            self.biases += b_slope*abs(self.biases)
