import numpy as np
from untitlednn.autodiff import tensor, Tensor
from untitlednn.nn import NeuralNetwork
from untitlednn.layer import Dense, ReLU, Dropout
from untitlednn.model import Model
from untitlednn.initializer import ZeroInitializer
from untitlednn.loss import SoftmaxCrossEntropy
from untitlednn.optimizer import Adam
from untitlednn.evaluator import OneHotAccEvaluator

data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype='float')
labels = np.array([0, 1, 2, 0])

# 分训练集、测试集
train_x = data[:3]
train_y = labels[:3]

test_x = data[3:]
test_y = labels[3:]

# 数据逐特征标准化

x_mean = train_x.mean(axis=0)
x_range = train_x.max(axis=0) - train_x.min(axis=0)

train_x -= x_mean
train_x /= x_range

test_x -= x_mean
test_x /= x_range


# 标签 one-hot 编码

def one_hot(targets, num_classes):
    return np.eye(num_classes)[targets.astype('int32').reshape(-1)]


# one-hot
train_y = one_hot(train_y, 3)
test_y = one_hot(test_y, 3)

# validation

validation_split_idx = 2

train_x = tensor(train_x[validation_split_idx:])
train_y = tensor(train_y[validation_split_idx:])

validation_x = tensor(train_x[:validation_split_idx])
validation_y = tensor(train_y[:validation_split_idx])

test_x = tensor(test_x)
test_y = tensor(test_y)

print(train_x.shape, validation_x.shape)

# construct network
network = NeuralNetwork([
    Dense(train_x.shape[1], 3, w_init=ZeroInitializer()),
    # Dense(10, 3, w_init=ZeroInitializer()),
])

# build model
model_unn = Model(network,
                  loss=SoftmaxCrossEntropy(),
                  optimizer=Adam(lr=1e-3),
                  evaluator=OneHotAccEvaluator)

model_unn.summary()

# fit
model_unn.fit(train_x, train_y,
              batch_size=1,
              epochs=10,
              validation_data=(validation_x, validation_y))

print('evaluate:', model_unn.evaluate(test_x, test_y))

for p, g in model_unn.nn.get_params_and_grads():
    if isinstance(p['w'], Tensor):
        print('p:', list(map(lambda x: x.value, p.values())))
    else:
        print('p:', p)
    if isinstance(g['w'], Tensor):
        print('g:', list(map(lambda x: x.value, g.values())))
    else:
        print('g:', g)
