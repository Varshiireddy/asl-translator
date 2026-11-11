import random
from micrograd.engine import Value


class Neuron:
    """One neuron: w·x + b, then ReLU (or nothing for the last layer)."""

    def __init__(self, n_in, relu=True):
        self.w = [Value(random.uniform(-1, 1) * (2 / n_in) ** 0.5) for _ in range(n_in)]
        self.b = Value(0.0)
        self.relu = relu

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.relu else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A list of neurons that all see the same input."""

    def __init__(self, n_in, n_out, relu=True):
        self.neurons = [Neuron(n_in, relu) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Layers stacked: MLP(42, [16, 5]) means 42 inputs -> 16 hidden -> 5 outputs."""

    def __init__(self, n_in, sizes):
        dims = [n_in] + sizes
        self.layers = [Layer(dims[i], dims[i + 1], relu=(i != len(sizes) - 1))
                       for i in range(len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
