import numpy as np

from untitlednn.initializer import RandomInitializer, ZeroInitializer
from untitlednn.autodiff import AutoDiff, tensor


class Layer(object):
    """Base Layer"""

    def __init__(self, name):
        self.name = name
        self.params = {}
        self.grads = {}
        self.inputs = None

        self.in_shape = None
        self.out_shape = None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grads):
        with AutoDiff(self.inputs) as ad:
            f = self.forward(self.inputs)

        g = ad.gradient(f, self.inputs, output_grad=grads)

        for key in self.params:
            # self.grads[key] = ad.gradient(f, self.params, output_grad=grads)
            # 👇下面这行代码等于上面的这行👆，减少函数调用
            self.grads[key] = self.params[key].grad

        return g

    @property
    def param_num(self):
        if not self.params:
            return 0
        num = 0
        for v in self.params.values():
            num += v.size

        return num


class Dense(Layer):
    def __init__(self,
                 num_in,
                 num_out,
                 w_init=RandomInitializer(),
                 b_init=ZeroInitializer()):
        super().__init__("Dense")

        self.in_shape = num_in
        self.out_shape = num_out

        self.params = {
            "w": w_init([num_in, num_out]),
            "b": b_init([1, num_out]),
        }

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    # def backward(self, grads):
    #     self.grads["w"] = self.inputs.T @ grads
    #     self.grads["b"] = np.sum(grads, axis=0)
    #     return grads @ self.params["w"].T


class Activation(Layer):
    """Base Activation Layer"""

    def __init__(self, name):
        super().__init__(name)
        self.inputs = None

    def func(self, x):
        raise NotImplementedError

    def derivative_func(self, x):
        raise NotImplementedError

    def forward(self, inputs):
        self.inputs = inputs
        return tensor(self.func(inputs))

    def backward(self, grads):
        return tensor(self.derivative_func(self.inputs)) * grads


class Sigmoid(Activation):
    def __init__(self):
        super().__init__("Sigmoid")

    def func(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_func(self, x):
        return self.func(x) * (1 - self.func(x))


class ReLU(Activation):
    def __init__(self):
        super().__init__("ReLU")

    def func(self, x):
        return np.maximum(x, 0)

    def derivative_func(self, x):
        return x > 0


class Dropout(Layer):
    def __init__(self, keep_prob=0.5):
        super().__init__("Dropout")
        self._keep_prob = keep_prob
        self._multiplier = None

    def forward(self, inputs):
        multiplier = np.random.binomial(
            1, self._keep_prob, size=inputs.shape)
        self._multiplier = multiplier / self._keep_prob
        outputs = inputs * self._multiplier
        return outputs

    def backward(self, grad):
        return grad * self._multiplier
