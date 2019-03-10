from numpy import array, round, exp, random, dot, transpose




def sigmoid(x, deri=False):
    if deri:
        return x * (1 - x)
    return 1 / (1 + exp(-x))


class NeuronLayer:
    def __init__(self, n1, n2, n3):
        self.w1 = 2 * random.random((n1, n2)) - 1
        self.w2 = 2 * random.random((n2, n3)) - 1

    def think(self, i1):
        return sigmoid(dot(sigmoid(dot(i1, self.w1)), self.w2))

    def learn(self, i1, y, times):
        for x in range(times):
            i3 = self.think(i1)
            i2 = sigmoid(dot(i1, self.w1))

            e2 = y - i3
            de2 = e2 * sigmoid(i3, True)

            e1 = de2.dot(transpose(self.w2))
            de1 = e1 * sigmoid(i2, True)
            self.w1 += i1.T.dot(de1)
            self.w2 += i2.T.dot(de2)
