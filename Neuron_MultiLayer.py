from numpy import array, round, exp, random, dot, transpose
random.seed(1)

def sigmoid(x, deri=False):
    if deri:
        return x * (1 - x)
    return 1 / (1 + exp(-x))


class NeuronMultiLayer:
    def __init__(self, n1, n2, n3, n4):
        self.w1 = 2 * random.random((n1, n2)) - 1
        self.w2 = 2 * random.random((n2, n3)) - 1
        self.w3 = 2 * random.random((n3, n4)) - 1

    def think(self, i1):
        return sigmoid(dot(sigmoid(dot(sigmoid(dot(i1, self.w1)), self.w2)), self.w3))

    def learn(self, i1, y, times):
        for x in range(times):
            i4 = self.think(i1)
            i3 = sigmoid(dot(sigmoid(dot(i1, self.w1)), self.w2))
            i2 = sigmoid(dot(i1, self.w1))

            e3 = y - i4
            print(sum(e3))
            de3 = e3 * sigmoid(i4, True)

            e2 = de3.dot(transpose(self.w3))
            de2 = e2 * sigmoid(i3, True)

            e1 = de2.dot(transpose(self.w2))
            de1 = e1 * sigmoid(i2, True)

            self.w1 += i1.T.dot(de1)
            self.w2 += i2.T.dot(de2)
            self.w3 += i3.T.dot(de3)
