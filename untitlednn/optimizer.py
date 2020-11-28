import numpy as np


class Optimizer(object):
    """Optimizer computes the steps to be added onto params, which optimizes the model
    """
    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def compute_steps(self, grads, params):
        """compute_steps 通过 grads 计算 params 的改变步长
        """
        # 展开梯度
        grads_flatten = np.concatenate([np.ravel(v) for grad in grads for v in grad.values()])
        # 计算每个参数的改变步长
        step_flatten = self._compute_steps(grads_flatten)
        # reshape, get step
        step = []
        offset = 0
        for param in params:
            layer = {}
            for k, v in param.items():
                length = np.prod(v.shape)
                _step = step_flatten[offset:offset + length].reshape(v.shape)
                _step -= self.weight_decay * v
                layer[k] = _step
                offset += length
            step.append(layer)

        return step

    def _compute_steps(self, grads):
        """给定一系列梯度，计算返回每个梯度对应的改变大小
        """
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)

        self._b1 = beta1
        self._b2 = beta2
        self._eps = eps

        self._t = 0
        self._m = 0
        self._v = 0

    def _compute_steps(self, grad):
        self._t += 1

        self._m = self._b1 * self._m + (1 - self._b1) * grad
        self._v = self._b2 * self._v + (1 - self._b2) * (grad ** 2)

        # bias correction
        _m = self._m / (1 - self._b1 ** self._t)
        _v = self._v / (1 - self._b2 ** self._t)

        step = -self.lr * _m / (_v ** 0.5 + self._eps)

        return step
