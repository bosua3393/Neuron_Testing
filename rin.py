from Neuron_Layer import NeuronLayer
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

train_output = array([[1], [0], [0], [1], [1], [1], [1], [1], [1], [0], [1], [0]])

test_data = array([[1, 0, 1, 0, 1, 1],
                   [1, 0, 0, 1, 0, 0],
                   [1, 1, 0, 1, 1, 0],
                   [0, 0, 1, 0, 1, 0],
                   [0, 1, 0, 1, 1, 1]])

test_output = array([[1], [1], [0], [0], [1]])


# ------------------------------------

rin = NeuronLayer(6, 6, 1)
rin.learn(train_data, train_output, 100000)


# ------------------------------------

print(rin.w1)
print(rin.w2)

print("Solved:")
print(rin.think(test_data))
print("error")
print(round((rin.think(test_data)-test_output)*100), sum(round((abs(rin.think(test_data)-test_output))*100)/5))
