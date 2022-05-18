import numpy as np
from grader import sigmoid, NeuronAbstract, ModelAbstract


class Softmax(NeuronAbstract):
    def __init__(self):
        pass  # Implement Here


class Sigmoid(NeuronAbstract):
    def __init__(self):
        pass  # Implement Here


class Affine(NeuronAbstract):
    def __init__(self, input_size, output_size):
        pass  # Implement Here


class Model(ModelAbstract):
    def forward(self, x):
        pass  # Implement Here

    def backward(self):
        pass  # Implement Here


def get_model():
    model = Model()
    # Implement Here
    return model
