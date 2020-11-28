import numpy as np
from untitlednn.tensor import Tensor


class Initializer(object):
    """Initializer initialize parameters of layers

    Example:
         initer = Initializer(shape)
         params = initer()
    """

    def init(self, shape):
        raise NotImplementedError

    def __call__(self, shape):
        return Tensor(self.init(shape).astype(np.float32))


class RandomInitializer(Initializer):
    """RandomInitializer initialize parameters of layers with random float in the interval (-0.05, 0.05)
    """

    def init(self, shape):
        return 0.1 * (np.random.random(shape) - 0.5)


class ZeroInitializer(Initializer):
    """ZeroInitializer initialize parameters of layers by zeros
    """

    def init(self, shape):
        return np.zeros(shape)
