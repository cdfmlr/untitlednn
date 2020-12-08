# Author: borgwang <borgwang@126.com>
# Date: 2018-05-23
#
# Filename: data_iterator.py
# Description: Data Iterator class


from collections import namedtuple

import numpy as np

from untitlednn.autodiff import tensor

Batch = namedtuple("Batch", ["inputs", "targets"])


class BaseIterator(object):

    def __call__(self, inputs, targets):
        raise NotImplementedError


class BatchIterator(BaseIterator):

    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs, targets):
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            idx = np.arange(len(inputs))
            np.random.shuffle(idx)
            inputs = inputs[idx]
            targets = targets[idx]

        for start in starts:
            end = start + self.batch_size
            batch_inputs = tensor(inputs[start: end])
            batch_targets = tensor(targets[start: end])

            lack = self.batch_size - len(batch_inputs)
            if lack != 0:
                if len(inputs) > self.batch_size:
                    batch_inputs = tensor(inputs[start - lack: end])
                    batch_targets = tensor(targets[start - lack: end])
                else:
                    raise ValueError('No enough data to generate any batch.')

            yield Batch(inputs=batch_inputs, targets=batch_targets)
