import warnings

import numpy as np

from untitlednn.initializer import RandomInitializer, ZeroInitializer
from untitlednn.autodiff import AutoDiff, tensor, Tensor, identity, Executor


class Layer(object):
    """Base Layer"""

    def __init__(self, name):
        self.name = name
        self.params = {}
        self.grads = {}
        self.inputs = None

        self.in_shape = None
        self.out_shape = None

        # for auto diff
        self.auto_diff_obj = None
        self.forward_output = None

        self.__param_num = None

    # @profile    # https://github.com/pythonprofilers/memory_profiler
    def forward_with_autodiff(self, inputs):
        """
        forward_with_autodiff 是对 forward 的自动微分封装
        """
        self.inputs = tensor(inputs)

        with AutoDiff(self.inputs) as ad:
            f = self.forward(self.inputs)

        self.auto_diff_obj = ad
        self.forward_output = f

        # A BED IMPLEMENT: return f
        # 这里要返回个新的 tensor, 断开与下一层的联系。
        # 不然后面的层把 f 作为输入, f (即这一层的 forward_output) 会被下一层的 AutoDiff
        # 置为 identity，然后当前层需要保留的计算图连接 (backward 计算梯度需要的
        # inputs -> ... -> forward_output ) 就丢失了。
        # 同时, 层与层直接不直接连接有助于内存优化, see Layer.backward_with_clean
        return tensor(f)

    def forward(self, inputs):
        """forward 是向前传播的具体算法

        通过重载来实现

        :param inputs: 输入值
        :return: 层的前向计算输出值
        """
        raise NotImplementedError

    # @profile    # https://github.com/pythonprofilers/memory_profiler
    def backward(self, grads):
        """backward 反向传播计算梯度。

        默认通过 forward 自动微分计算，重载来自定义。

        :param grads: 输出（下一层）的梯度
        :return: 这一层的梯度
        """
        g = self.auto_diff_obj.gradient(self.forward_output, self.inputs, output_grad=grads)
        g = tensor(g)

        for key in self.params:
            # self.grads[key] = self.auto_diff_obj.gradient(self.forward_output, self.params[key], output_grad=grads)
            # 👇下面这行代码等于上面的这行👆，减少函数调用
            self.grads[key] = tensor(self.params[key].grad)

        # 断开计算图的连接
        self.auto_diff_obj.close()

        return g

    # @profile    # https://github.com/pythonprofilers/memory_profiler
    def backward_with_clean(self, grads):
        """Deprecated

        调用 self.backward 然后做内存清理工作。

        这个功能在 Layer.backward 中实现了, 因此不再使用此方法
        """
        warnings.warn("backward_with_clean is deprecated. "
                      "这个功能在 Layer.backward 中实现了, 因此不再使用此方法",
                      DeprecationWarning)

        g = tensor(self.backward(tensor(grads)))

        # # 0. clean up: of no avail
        # stack = [self.forward_output]
        # while stack:
        #     i = stack.pop(0)
        #     # print(len(stack), id(i), type(i), i.id if isinstance(i, Tensor) else -1, sys.getrefcount(i))
        #     if not isinstance(i, Tensor):
        #         del i
        #         continue
        #     elif i.op == identity:
        #         continue
        #     stack.extend(i.inputs)
        #     del i
        # # print('----')

        # # 1. del & call gc: useless
        # del self.inputs.grad
        # del self.forward_output
        # del self.auto_diff_obj
        #
        # n = gc.collect()
        # if n:
        #     print('gc:', n)

        # # 2. THIS WORKS!
        # # 断开计算图的连接: 这个是内存优化的关键！！！
        # # 计算图节点 (即 Tensor) 相互引用, 造成这些不再使用的节点, 无法及时被 GC 清理,
        # # 这些无法再次使用的计算图节点, 会长时间驻留内存, 直到 model.fit 的 for epoch
        # # 训练循环完全结束。
        # # 这个内存泄露问题会导致内存占用超出预期数十倍, 用 schoolwork 训练 10 轮作为比
        # # 较, 下面的代码可以使内存峰值从 3GiB 降低到 120 MiB。
        # ex = Executor(self.forward_output)
        # # print(len(ex.topo_list))
        # for i in ex.topo_list:
        #     if isinstance(i, Tensor):
        #         i.inputs = []

        # 把方法 2 封装到下层, 得到最终实现: 断开计算图的连接
        self.auto_diff_obj.close()

        return g

    @property
    def param_num(self):
        if not self.__param_num:
            if not self.params:
                self.__param_num = 0
            num = 0
            for v in self.params.values():
                num += v.size

            self.__param_num = num
        return self.__param_num


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

    # @profile    # https://github.com/pythonprofilers/memory_profiler
    def forward(self, inputs):
        # self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    # Do not override backward, use Layer.backward (auto diff)
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
        # self.inputs = inputs
        return tensor(self.func(inputs))

    def backward(self, grads):
        # self.grads['df'] = tensor(self.derivative_func(self.inputs)) * grads
        # return self.grads['df']
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
        return tensor(outputs)

    def backward(self, grad):
        # self.grads['grad'] = grad * self._multiplier
        # return self.grads['grad']
        return tensor(grad * self._multiplier)
