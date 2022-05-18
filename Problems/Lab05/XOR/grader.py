import numpy as np


class NeuronAbstract:
    def __init__(self):
        raise Exception("Not implemented constructor")

    def forward(self, x):
        raise Exception("Not implemented forward")

    def loss(self, t):
        raise Exception("Not implemented loss")

    def backward(self, d_out, lr):
        raise Exception("Not implemented backward")


class ModelAbstract:
    def __init__(self, learning_rate):
        self.layer = []
        self.error = None
        self.learning_rate = learning_rate

    def add(self, layer):
        self.layer.append(layer)

    def getLoss(self, t):
        self.error = self.layer[-1].loss(t)
        return self.error

    def forward(self, x):
        raise Exception("Not implemented forward")

    def backward(self):
        raise Exception("Not implemented backward")


def sigmoid(x):
    np.clip(x, -100, 100, out=x)
    return 1 / (1 + np.exp((-1) * x))


def mean_squared_error(y, t):
    return np.mean((y - t) ** 2)


if __name__ == '__main__':
    from xor import get_model

    nb_epochs = 1000

    x_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])
    model = get_model()

    for epoch in range(nb_epochs):
        hypothesis = model.forward(x_train)
        model.getLoss(y_train)
        model.backward()
        print("Epoch : %d/%d, loss : %.7f" % (epoch + 1, nb_epochs, np.mean(model.getLoss(y_train) ** 2)))

    print('hypothesis', hypothesis)
    print('loss', model.getLoss(y_train))
