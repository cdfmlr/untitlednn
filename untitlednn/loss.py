import numpy as np


class Loss(object):
    """Loss 计算损失
    """
    def loss(self, predicted, actual):
        raise NotImplementedError

    def grad(self, predicted, actual):
        raise NotImplementedError


class MSELoss(Loss):
    def loss(self, predicted, actual):
        m = predicted.shape[0]
        return np.sum((predicted - actual) ** 2) / (2 * m)

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        return (predicted - actual) / m


class SoftmaxCrossEntropyLoss(Loss):
    def loss(self, predicted, actual):
        m = predicted.shape[0]

        # softmax
        exps = np.exp(predicted - np.max(predicted))
        p = exps / np.sum(exps)

        # cross entropy loss
        nll = -np.log(np.sum(p * actual, axis=1))

        return np.sum(nll) / m

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        grads = predicted - actual
        return grads / m
