import argparse
import time

import numpy as np
from tensorflow.keras.datasets import mnist  # to get MNIST conveniently

from untitlednn.tensor import Tensor
from untitlednn.nn import NeuralNetwork
from untitlednn.layer import Dense, ReLU
from untitlednn.model import Model
from untitlednn.loss import SoftmaxCrossEntropyLoss
from untitlednn.optimizer import Adam
from untitlednn.evaluator import OneHotAccEvaluator, MSEEvaluator
from untitlednn.util.data_iterator import BatchIterator

from untitlednn.initializer import RandomInitializer, ZeroInitializer

MNIST_IMG_LEN = 28 * 28


def one_hot(targets, num_classes):
    return np.eye(num_classes)[Tensor(targets).reshape(-1)]


def prepare_mnist():
    print("Loading MNIST...", end="")
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    print("Done.")

    print("Prepare data...", end="")
    train_x = train_x.reshape((60000, MNIST_IMG_LEN)).astype('float32') / 255
    train_x = Tensor(train_x)

    test_x = test_x.reshape((10000, MNIST_IMG_LEN)).astype('float32') / 255
    test_x = Tensor(test_x)

    train_y = Tensor(one_hot(train_y, 10))
    test_y = Tensor(one_hot(test_y, 10))

    print("Done.")

    return (train_x, train_y), (test_x, test_y)


def main(args):
    (train_x, train_y), (test_x, test_y) = prepare_mnist()

    network = NeuralNetwork([
        Dense(MNIST_IMG_LEN, 200),
        ReLU(),
        Dense(200, 100),
        ReLU(),
        Dense(100, 70),
        ReLU(),
        Dense(70, 30),
        ReLU(),
        Dense(30, 10),
    ])

    model = Model(network,
                  loss=SoftmaxCrossEntropyLoss(),
                  optimizer=Adam(lr=args.lr),
                  evaluator=OneHotAccEvaluator)
    model.summary()

    print("Start train:")
    model.fit(train_x, train_y, batch_size=args.batch_size, epochs=args.num_epoch, validation_data=(test_x, test_y))
    model.save("/Users/c/Desktop/mnist.pkl")

    # iterator = BatchIterator(batch_size=args.batch_size)
    # losses = []
    # for epoch in range(args.num_epoch):
    #     # train epoch
    #     t_start = time.time()
    #
    #     for batch in iterator(train_x, train_y):
    #         pred = model.forward(batch.inputs)
    #         loss, grads = model.loss_bp_grads(pred, batch.targets)
    #         model.apply_grads(grads)
    #         losses.append(loss)
    #
    #     t_end = time.time()
    #
    #     # evaluate epoch
    #
    #     test_pred = model.forward(test_x)
    #     res = model.evaluator.evaluate(test_pred, test_y)
    #
    #     print(f"Epoch {epoch + 1}/{args.num_epoch}\t{t_end - t_start :.4f}s\t{res}")


if __name__ == "__main__":
    # shape = (5, 5)
    # print(XavierUniformInit()(shape))
    # print(RandomInitializer()(shape))
    # print(ZeroInitializer()(shape))
    # exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", default=5, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    args = parser.parse_args()
    main(args)