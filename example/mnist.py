import argparse
import time

import numpy as np
from tensorflow.keras.datasets import mnist  # to get MNIST conveniently

from untitlednn.autodiff import tensor
from untitlednn.nn import NeuralNetwork
from untitlednn.layer import Dense, ReLU
from untitlednn.model import Model
from untitlednn.loss import SoftmaxCrossEntropy
from untitlednn.optimizer import Adam, SGD
from untitlednn.evaluator import OneHotAccEvaluator
from untitlednn.util.data_iterator import BatchIterator

MNIST_IMG_LEN = 28 * 28
MODEL_SAVE_PATH = "/Users/c/Desktop/mnist.pkl"


def one_hot(targets, num_classes):
    return np.eye(num_classes)[tensor(targets).reshape(-1)]


def prepare_mnist():
    """Load and prepare the MNIST dataset

    :return: (train_x, train_y), (test_x, test_y)
    """
    print("Loading MNIST...", end="")
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    print("Done.")

    print("Prepare data...", end="")
    train_x = train_x.reshape((60000, MNIST_IMG_LEN)).astype('float32') / 255
    train_x = tensor(train_x)

    test_x = test_x.reshape((10000, MNIST_IMG_LEN)).astype('float32') / 255
    test_x = tensor(test_x)

    train_y = tensor(one_hot(train_y, 10))
    test_y = tensor(one_hot(test_y, 10))

    print("Done.")

    return (train_x, train_y), (test_x, test_y)


def main(args):
    # prepare data
    (train_x, train_y), (test_x, test_y) = prepare_mnist()

    # construct network
    network = NeuralNetwork([
        Dense(MNIST_IMG_LEN, 100),
        ReLU(),
        Dense(100, 30),
        ReLU(),
        Dense(30, 10),
    ])

    # build model
    model = Model(network,
                  loss=SoftmaxCrossEntropy(),
                  optimizer=Adam(lr=args.lr),
                  evaluator=OneHotAccEvaluator)
    model.summary()

    # train the model
    print("Start train:")
    model.fit(train_x, train_y, batch_size=args.batch_size, epochs=args.num_epoch, validation_data=(test_x, test_y))

    # save and load
    print(f"Save model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

    print(f"Load model from {MODEL_SAVE_PATH}.")
    model: Model = Model.load(MODEL_SAVE_PATH)

    # evaluate the model
    print("Evaluate on train set:", end='\t')
    print(model.evaluate(train_x, train_y))

    print("Evaluate on  test set:", end='\t')
    print(model.evaluate(test_x, test_y))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", default=5, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)  # Adam: 1e-3, SGD: 1e-1
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()
    main(args)
