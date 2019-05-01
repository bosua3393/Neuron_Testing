from Neuron import Neuron
from numpy import array, round

train_data = array([[1, 0, 1, 1, 0, 0],
                    [1, 1, 1, 0, 1, 0],
                    [1, 1, 1, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0, 1],
                    [0, 1, 0, 1, 1, 0],
                    [0, 1, 1, 0, 1, 0],
                    [1, 0, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 1, 1]])

train_output = array([[1], [1], [1], [1], [0], [1], [0], [0], [1], [1], [0], [0]])

test_data = array([[1, 0, 1, 0, 1, 1],
                   [1, 0, 0, 1, 0, 0],
                   [0, 1, 0, 1, 1, 0],
                   [1, 0, 1, 0, 1, 0],
                   [0, 1, 0, 1, 1, 1]])

test_output = array([[1], [1], [0], [1], [0]])


yui = Neuron(6, 1)
yui.learn(train_data, train_output, 10000)

print(yui.weights, yui.biases)
print("Solving .33...")
print(round(yui.think(test_data)))
print("%error:")
print(round((test_output-yui.think(test_data))*100))