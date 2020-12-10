import json
import os
import pickle
import time
import numpy as np
from pympler import tracker

from untitlednn.autodiff import Tensor
from untitlednn.util.data_iterator import BatchIterator


class Model(object):
    """Model packs NeuralNetwork with the Loss, Optimizer and Evaluator.
    """

    def __init__(self, nn, loss, optimizer, evaluator, **kwargs):
        self.nn = nn
        self.loss = loss
        self.optimizer = optimizer
        self.evaluator = evaluator

        self.name = kwargs.get("name", f"Model_{id(self)}")

    # @profile    # https://github.com/pythonprofilers/memory_profiler
    def forward(self, inputs):
        return self.nn.forward(inputs)

    def backward(self, grad):
        return self.nn.backward(grad)

    # @profile    # https://github.com/pythonprofilers/memory_profiler
    def loss_bp_grads(self, predictions, targets) -> (Tensor, Tensor):
        """loss_bp_grads 计算 predictions 对于 targets 的损失，反向传播得到梯度，调用优化器得到每个参数待更新的步长。
        :returns: (grads, steps)
        """
        losses = self.loss.loss(predictions, targets)
        loss_grad = self.loss.grad(predictions, targets)

        grads = self.nn.backward(loss_grad)
        params = self.nn.get_params()
        steps = self.optimizer.compute_steps(grads, params)

        return losses, steps

    def apply_grads(self, grads) -> None:
        """apply_grads 给网络中的每个参数加上 grads
        """
        for grad, (param, _) in zip(grads, self.nn.get_params_and_grads()):
            for k in param.keys():
                param[k] += grad[k]

    # @profile    # https://github.com/pythonprofilers/memory_profiler
    def fit(self, train_x, train_y, batch_size=128, epochs=5, validation_data=None, verbose=True) -> None:
        """fit 在数据集 (train_x, train_y) 上训练模型。

        :param train_x: Tensor, 数据
        :param train_y: Tensor, 真实结果
        :param batch_size: int, 批的大小
        :param epochs: int, 训练轮次
        :param validation_data: 验证集 (val_x, val_y)
        :param verbose: 打印训练过程

        :return: None

        TODO: returns history
        """
        # tr = tracker.SummaryTracker()    # memory debug
        iterator = BatchIterator(batch_size=int(batch_size))

        losses = []
        for epoch in range(int(epochs)):
            # train epoch
            tt_start = time.time()
            for batch in iterator(train_x, train_y):
                pred = self.forward(batch.inputs)
                loss, grads = self.loss_bp_grads(pred, batch.targets)
                self.apply_grads(grads)
                losses.append(loss)
            tt_end = time.time()

            # evaluate epoch
            te_start = time.time()
            if validation_data:
                res = self.evaluate(validation_data[0], validation_data[1])
                if not res.get("loss", None):
                    res["loss"] = float(losses[-1])
            else:
                res = {"loss": float(losses[-1])}
            te_end = time.time()

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}:\t"
                      f"train {tt_end - tt_start :.4f}s, evaluate {te_end - te_start :.4f}s\t"
                      f"{res}")
            # time.sleep(0.5)    # for debug
        # tr.print_diff()    # memory debug

    def evaluate(self, test_x, test_y) -> dict:
        """evaluate 预测 test_x 的结果，评估模型在数据集 (test_x, test_y) 上的表现,
        返回 Evaluator 评估的结果 —— a dict.
        """
        pred = self.forward(test_x)
        return self.evaluator.evaluate(pred, test_y)

    def summary(self):
        s = f'Model: "{self.name}"\n'

        total_param_num = 0

        layers = [("Layer", "OutputShape", "Param#")]
        for layer in self.nn.layers:
            total_param_num += layer.param_num
            layers.append(map(str, (layer.name, layer.out_shape, layer.param_num)))

        s += '_' * 28 + '\n'
        s += '\t'.join(layers.pop(0)) + '\n'
        s += '=' * 28 + '\n'
        for ly in layers:
            s += '\t\t'.join(ly) + '\n'
        s += '=' * 28 + '\n'
        s += f"Trainable Params: {total_param_num}\n"

        print(s)

        return s

    def save(self, file_path) -> None:
        """Save model to file_path
        """
        if os.path.exists(file_path):
            assert os.path.isfile(file_path)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        """Load saved model (saved by Model.save) from file_path

        :return: Model object
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)
