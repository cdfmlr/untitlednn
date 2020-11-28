import numpy as np


class Tensor(np.ndarray):
    """Tensor is the np.ndarray

    Calling Tensor(array_like) to get a Tensor object.
    """

    def __new__(cls, array_like, *args, **kwargs):
        """Makes a Tensor object from an array_like.

        Refer:
        https://stackoverflow.com/questions/27557029/how-should-a-class-that-inherits-from-numpy-ndarray-and-has-a-default-value-be-c
        https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
        """
        obj = np.asarray(array_like).view(cls)
        return obj
