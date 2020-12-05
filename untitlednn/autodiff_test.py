from untitlednn.tensor import Tensor
from untitlednn.autodiff import *


def d_sin(inputs):
    return np.cos(inputs[0])


@differentiable(df=d_sin)
def sin1(inputs):
    return np.sin(inputs[0])


def test_autodiff():
    add = Add()
    sin = Sin()
    sin2 = Differentiable(np.sin)

    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])

    c = mul(sin(a), sin1(b))

    ex = Executor(c)
    print(ex.run())

    ex.grad()
    print(a.grad, b.grad)


def test_tensor():
    set_eager(False)

    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])

    # c = a + b
    c = sin1(a) * sin1(b)

    print('c:', c)
    print('c.value:', c.value)

    ex = Executor(c)
    print('ex.run(): ', ex.run())

    ex.grad()
    print('grads:', a.grad, b.grad)


def test_tensor_eager():
    set_eager(True)

    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])

    # c = a + b
    c = sin1(a) * sin1(b)

    print('c:', c)
    print('c.value:', c.value)

    # ex = Executor(c)
    # print('ex.run(): ', ex.run())
    #
    # ex.grad()
    print('grads:', a.grad, b.grad)


if __name__ == '__main__':
    # test_autodiff()
    test_tensor()
    test_tensor_eager()
