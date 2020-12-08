import pandas as pd
import numpy as np

# 导入数据
full_df = pd.DataFrame()
for i in range(1, 5):
    df = pd.read_csv(f'data/data{i}.csv', header=None)
    full_df = full_df.append(df)

# 打乱顺序
shuffled_df = full_df.sample(frac=1.0)

# 分离数据和标签
labels = shuffled_df.values[:, 0]
data = shuffled_df.values[:, 1:]

# 分训练集、测试集
train_x = data[:1500]
train_y = labels[:1500]

test_x = data[1500:]
test_y = labels[1500:]

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


# [1, 4] => [0, 3]
train_y -= 1
test_y -= 1

# one-hot
train_y = one_hot(train_y, 4)
test_y = one_hot(test_y, 4)

# Tensorflow 测试
#
# from tensorflow.keras.models import Model
# from tensorflow.keras import Input
# from tensorflow.keras import layers
#
#
# def build_model():
#     input_tensor = Input(shape=(train_x.shape[1],))
#     x = layers.Dense(256, activation='relu')(input_tensor)
#     x = layers.Dense(256, activation='relu')(x)
#     x = layers.Dropout(0.2)(x)
#     output_tensor = layers.Dense(4, activation='softmax')(x)
#
#     model = Model(input_tensor, output_tensor)
#
#     model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
#
#     return model
#
#
# model_tf = build_model()
# model_tf.summary()
#
# history = model_tf.fit(train_x, train_y,
#                        validation_split=0.1,
#                        epochs=10, batch_size=128)
#
# print("model_tf.evaluate:")
# model_tf.evaluate(test_x, test_y, batch_size=128)

# UntitledNN

from untitlednn.autodiff import tensor
from untitlednn.nn import NeuralNetwork
from untitlednn.layer import Dense, ReLU, Dropout
from untitlednn.model import Model
from untitlednn.loss import SoftmaxCrossEntropy
from untitlednn.optimizer import Adam
from untitlednn.evaluator import OneHotAccEvaluator

# validation
validation_split = 0.1
validation_split_idx = int(validation_split * train_x.shape[0])

train_x = tensor(train_x[validation_split_idx:])
train_y = tensor(train_y[validation_split_idx:])

validation_x = tensor(train_x[:validation_split_idx])
validation_y = tensor(train_y[:validation_split_idx])

test_x = tensor(test_x)
test_y = tensor(test_y)

print(train_x.shape, validation_x.shape)

# construct network
network = NeuralNetwork([
    Dense(train_x.shape[1], 256),
    ReLU(),
    Dense(256, 256),
    ReLU(),
    Dropout(0.8),
    Dense(256, 4),
])

# build model
model_unn = Model(network,
                  loss=SoftmaxCrossEntropy(),
                  optimizer=Adam(lr=1e-3),
                  evaluator=OneHotAccEvaluator)

model_unn.summary()

# fit
model_unn.fit(train_x, train_y,
              batch_size=128,
              epochs=1,
              validation_data=(validation_x, validation_y))

print('evaluate:', model_unn.evaluate(test_x, test_y))
