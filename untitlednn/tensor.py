# import numpy as np
# from untitlednn.autodiff import *
#
#
# class Tensor(Node):
#     """Tensor is a autodiff Node with identity Op with only one value
#
#     Calling Tensor(array_like) to get a Tensor object.
#     """
#
#     def __init__(self, array_like):
#         self.data = np.asarray(array_like)
#         Node.__init__(self, op=identity, inputs=[self.data])
