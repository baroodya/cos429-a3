"""
Basic script to create a new network model.
The model presented here is meaningless, but it shows how to properly 
call init_model and init_layers for the various layer types.
"""

import sys
import matplotlib.pyplot as plt
import copy

sys.path += ["initial"]
from data_utils import get_CIFAR10_data

from train import get_accuracy, train

sys.path += ["layers"]
import numpy as np
from init_layers import init_layers
from init_model import init_model
from inference import inference
from layers.loss_euclidean import loss_euclidean

from testing import test_implementations


def _init_model():
    l = [
        init_layers(
            "conv",
            {"filter_size": 3, "filter_depth": 3, "num_filters": 3},
        ),
        init_layers("relu", {}),
        init_layers(
            "conv",
            {"filter_size": 3, "filter_depth": 3, "num_filters": 3},
        ),
        init_layers("relu", {}),
        init_layers("pool", {"filter_size": 2, "stride": 2}),
        init_layers(
            "conv",
            {"filter_size": 3, "filter_depth": 3, "num_filters": 3},
        ),
        init_layers("relu", {}),
        init_layers(
            "conv",
            {"filter_size": 3, "filter_depth": 3, "num_filters": 3},
        ),
        init_layers("relu", {}),
        init_layers("pool", {"filter_size": 2, "stride": 2}),
        init_layers("flatten", {}),
        init_layers("linear", {"num_in": 75, "num_out": 10}),
        init_layers("softmax", {}),
    ]

    model = init_model(l, [32, 32, 3], 10, True)

    return model


def main():
    X_train, y_train, X_test, y_test = get_CIFAR10_data()

    model = _init_model()
    ref_model = copy.deepcopy(model)

    numIters = 500

    learning_rate = 0.01
    weight_decay = 0.0005
    batch_size = 256

    epsilon = 0

    save_file = "experiment1.npz"

    params = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "save_file": save_file,
        "X_test": X_test,
        "y_test": y_test,
        "epsilon": epsilon,
    }

    test_implementations(model, ref_model, X_train, y_train, params)

    (
        finished_model,
        loss,
        training_accuracies,
        testing_accuracies,
    ) = train(model, X_train, y_train, params, numIters)

    print(loss)

    plt.plot(training_accuracies)
    plt.xlabel("Iterations")
    plt.ylabel("Testing Accuracy")
    plt.show()

    final_test_accuracy = testing_accuracies[-1]


if __name__ == "__main__":
    main()
