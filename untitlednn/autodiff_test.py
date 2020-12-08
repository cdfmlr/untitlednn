# from untitlednn.tensor import Tensor
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

    a = tensor([1, 2, 3])
    b = tensor([4, 5, 6])

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

    a = tensor([0, 2, 3])
    b = tensor([4, 5, 6])

    a[0] = 1

    # c = a + b
    with AutoDiff(a, b) as ad:
        c = sin1(a) * sin1(b)

    da = ad.gradient(c, a)
    db = ad.gradient(c, b)

    print('c:', c, '\nc.shape:', c.shape)
    print('c.value:', c.value, 'c[0]:', c[0])

    print('grads:', a.grad, b.grad)
    print('da,db:', da, db)


if __name__ == '__main__':
    # test_autodiff()
    # test_tensor()
    test_tensor_eager()
