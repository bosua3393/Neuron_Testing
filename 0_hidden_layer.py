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


model = Neuron(6, 1)
model.learn(train_data, train_output, 10000)

print(model.weights, model.biases)
print("Solving .33...")
print(round(model.think(test_data)))
print("%error:")
print(round((test_output-model.think(test_data))*100))
