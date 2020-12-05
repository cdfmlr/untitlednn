import numpy as np
from functools import wraps

# 调试模式, 打印自动微分执行的信息
DEBUG = True
# 及时执行模式, 及时求值
EAGER = True


def set_eager(eager: bool):
    """设置及时执行模式
    """
    global EAGER
    EAGER = eager


class Node(object):
    """计算图的节点, 表示某个计算的结果
    """
    _global_id = 0  # TODO: Threading safe

    def __init__(self, op=None, inputs=None):
        self.inputs = inputs
        self.op = op

        self.id = Node._global_id
        Node._global_id += 1

        self.value = 0.0
        self.grad = 0.0

        if EAGER:  # 立即求值
            self.evaluate()
            ex = EagerExecutor(self)
            ex.grad()
            # self.evaluate()  # 立即求值
            if DEBUG:
                print(f"eager exec: {self}")

    def values_of_inputs(self) -> list:
        """输入的值
        :return: list of values
        """
        values = []
        for i in self.inputs:
            if isinstance(i, Node):
                i = i.value
            values.append(i)
        return values

    def evaluate(self) -> None:
        """计算当前节点的值
        :return: None
        """
        self.value = self.op.compute(self.values_of_inputs())

    # def __repr__(self):
    #     return self.__str__

    def __str__(self):
        return f'{self.op} {self.values_of_inputs()} = {self.value}, grad: {self.grad}'

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return add(self, -other)

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)


class Op(object):
    """Op 是各种运算的父类

    每次调用 Op 都会产生一个 Node. Op 本身不包含状态, 计算的状态保存在 Node 中。
    """

    def __call__(self, *inputs):
        """产生一个新的Node, 表示计算的结果
        """
        raise NotImplementedError

    def compute(self, inputs):
        """计算运算结果

        :param inputs: 输入数据
        :return: 计算结果
        """
        raise NotImplementedError

    def gradient(self, inputs, output_grad):
        """计算梯度

        :param inputs: 输入数据
        :param output_grad: 输出节点的梯度
        :return: 这个计算的梯度
        """
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return self.name


class Identity(Op):
    """这个计算的结果就是输入本身
    """

    def __call__(self, a):
        return Node(self, [a])

    def compute(self, inputs):
        return inputs[0]

    def gradient(self, inputs, output_grad):
        return [output_grad]


identity = Identity()


class Add(Op):
    """加"""

    def __call__(self, a, b):
        return Node(self, [a, b])

    def compute(self, inputs):
        return inputs[0] + inputs[1]

    def gradient(self, inputs, output_grad):
        return [output_grad, output_grad]


add = Add()


class Mul(Op):
    """乘"""

    def __call__(self, a, b):
        return Node(self, [a, b])

    def compute(self, inputs):
        return inputs[0] * inputs[1]

    def gradient(self, inputs, output_grad):
        return [inputs[1] * output_grad, inputs[0] * output_grad]


mul = Mul()


class Div(Op):
    """除"""

    def __call__(self, a, b):
        return Node(self, [a, b])

    def compute(self, inputs):
        return inputs[0] / inputs[1]

    def gradient(self, inputs, output_grad):
        v2 = inputs[1] ** 2
        return [inputs[1] * output_grad / v2, inputs[0] * output_grad / v2]


div = Div()


class Function(Op):
    """
    Function 是抽象的函数。实现了默认的数值微分。
    """

    def __call__(self, *inputs):
        return Node(self, inputs)

    def f(self, inputs):
        """具体的函数

        :param inputs: 输入列表
        :return: 函数的输出值
        """
        raise NotImplementedError

    def df(self, inputs):
        """f 的导函数，默认实现是做数值微分

        :param inputs: 输入列表
        :return: 微分值
        """
        # 数值微分
        h = self.h(inputs)
        return (self.compute([i + h for i in inputs]) - self.compute([i - h for i in inputs])) / (2 * h)

    def compute(self, inputs):
        return self.f(inputs)

    def gradient(self, inputs, output_grad):
        """函数的微分计算，默认做数值微分
        """
        return [self.df(inputs) * output_grad]

    def h(self, inputs):
        """计算数值微分的步长。

        这里的出现透明实现是一个常数。但这个函数传入了输入的值，可以重载来根据输入计算步长。

        :param inputs: 输入，用来帮助动态确定步长
        :return: 步长 h
        """
        return 1e-4


class Sin(Function):
    def f(self, inputs):
        return np.sin(inputs[0])

    def df(self, inputs):
        return np.cos(inputs[0])


class Differentiable(Function):
    """
    Differentiable 是一个装饰器，把函数变成可自动微分的。

    默认使用数值微分：

    ```
    differentiable_sin = Differentiable(np.sin)
    ```

    也可以自己写导函数：

    ```
    def d_sin(x):
        return np.cos(x)

    differentiable_sin = Differentiable(np.sin, df=d_sin)
    ```
    """

    def __init__(self, f, df=None):
        self.f = f
        if df:
            self.df = df


def differentiable(df=None):
    """
    differentiable 是 Differentiable 的函数装饰器版本，把函数变成可自动微分的。

    默认使用数值微分：

    ```
    @differentiable
    def sin(x):
        return np.sin(x)
    ```

    也可以自己写导函数：

    ```
    def d_sin(x):
        return np.cos(x)

    @differentiable(df=d_sin)
    def another_sin(x):
        return np.sin(x)
    ```
    """

    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return Differentiable(f, df=df)(*args, **kwargs)

        return wrapper

    return decorate


class Executor(object):
    """计算图的执行和自动微分
    """

    def __init__(self, root):
        self.topo_list = self._topo_sort(root)
        self.root = root

    def run(self):
        """前向执行计算图

        :return: 计算的结果
        """
        node_evaluated = set()
        for n in self.topo_list:
            if id(n) not in node_evaluated:
                n.evaluate()
                node_evaluated.add(id(n))
                if DEBUG:
                    print("evaluate:", n)

        return self.root.value

    def _dfs(self, lst, node):
        if not isinstance(node, Node):
            return

        for n in node.inputs:
            self._dfs(lst, n)

        lst.append(node)

    def _topo_sort(self, root):
        lst = []
        self._dfs(lst, root)
        return lst

    def grad(self, output_grad=1.0) -> None:
        """反向计算梯度，结果在每个节点中

        :param output_grad: 输出节点的梯度
        :return: None
        """
        self.topo_list[-1].grad = output_grad  # 输出节点
        for n in reversed(self.topo_list):
            grad = n.op.gradient(n.values_of_inputs(), n.grad)
            for i, g in zip(n.inputs, grad):
                if isinstance(i, Node):
                    i.grad += g
        if DEBUG:
            print("\nAUTODIFF:")
            for n in reversed(self.topo_list):
                print(n)
            print("-----\n")


class EagerExecutor(Executor):
    """及时执行模式下用这个计算微分

    仅限内部使用! 不应该被外部调用。

    及时执行模式下，每一次计算都会及时求微分。每一次求微分时不应该累计上一次计算结果，
    而 Executor 会叠加计算。
    所以这个 EagerExecutor 在遍历节点的时候会把之前计算出的 grad 置为 0。
    """
    def _dfs(self, lst, node):
        super()._dfs(lst, node)
        if isinstance(node, Node):
            node.grad = 0.0  # reset grad
