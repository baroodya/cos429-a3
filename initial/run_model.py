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
    # l = [
    #     init_layers(
    #         "conv",
    #         {"filter_size": 3, "filter_depth": 3, "num_filters": 3},
    #     ),
    #     init_layers("relu", {}),
    #     init_layers(
    #         "conv",
    #         {"filter_size": 3, "filter_depth": 3, "num_filters": 3},
    #     ),
    #     init_layers("relu", {}),
    #     init_layers("pool", {"filter_size": 2, "stride": 2}),
    #     init_layers(
    #         "conv",
    #         {"filter_size": 3, "filter_depth": 3, "num_filters": 3},
    #     ),
    #     init_layers("relu", {}),
    #     init_layers(
    #         "conv",
    #         {"filter_size": 3, "filter_depth": 3, "num_filters": 3},
    #     ),
    #     init_layers("relu", {}),
    #     init_layers("pool", {"filter_size": 2, "stride": 2}),
    #     init_layers("flatten", {}),
    #     init_layers("linear", {"num_in": 75, "num_out": 10}),
    #     init_layers("softmax", {}),
    # ]
    l = [
        init_layers(
            "conv",
            {"filter_size": 2, "filter_depth": 3, "num_filters": 3},
        ),
        init_layers("relu", {}),
        init_layers(
            "conv",
            {"filter_size": 2, "filter_depth": 3, "num_filters": 3},
        ),
        init_layers("pool", {"filter_size": 2, "stride": 2}),
        init_layers("relu", {}),
        init_layers("flatten", {}),
        init_layers("linear", {"num_in": 675, "num_out": 10}),
        init_layers("softmax", {}),
    ]

    model = init_model(l, [32, 32, 3], 10, True)

    return model


def main():
    X_train, y_train, X_test, y_test = get_CIFAR10_data()

    model = _init_model()

    numIters = 10000

    learning_rate = 1e-5
    weight_decay = 0.005
    batch_size = 128

    rho = 0.99

    velocity = []

    save_file = "experiment1.npz"

    params = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "save_file": save_file,
        "X_test": X_test,
        "y_test": y_test,
        "rho": rho,
        "velocity": velocity,
    }

    (
        duration,
        training_loss,
        testing_loss,
        training_accuracies,
        testing_accuracies,
    ) = train(model, X_train, y_train, params, numIters)

    print(
        "Final Testing Accuracy = ",
        testing_accuracies[-1],
        "\n",
        "Time per iteration: ",
        duration / numIters,
        "\n",
        "Total time: ",
        duration,
        sep="",
    )

    # training_loss = range(1000)
    # testing_loss = range(1000)

    plt.title("Model 2")
    plt.subplot(1, 2, 1)
    plt.plot(training_loss)
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(testing_loss)
    plt.xlabel("Iterations")
    plt.ylabel("Testing Loss")
    plt.show()


if __name__ == "__main__":
    main()
