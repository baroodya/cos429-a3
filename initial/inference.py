import numpy as np

import sys

sys.path += ["layers"]
sys.path += ["pyc_code"]

from layers.fn_conv import fn_conv
from layers.fn_flatten import fn_flatten
from pyc_code.fn_linear_ import fn_linear
from layers.fn_pool import fn_pool
from layers.fn_relu import fn_relu
from pyc_code.fn_softmax_ import fn_softmax


def inference(model, input):
    """
    Do forward propagation through the network to get the activation
    at each layer, and the final output
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
    Returns:
        output: The final output of the model
        activations: A list of activations for each layer in model["layers"]
    """

    num_layers = len(model["layers"])
    activations = [
        None,
    ] * num_layers

    # TODO: FORWARD PROPAGATION CODE
    layers = model["layers"]
    curr_input = input
    for i in range(num_layers):

        def conv():
            return fn_conv(
                curr_input,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=False,
            )

        def flatten():
            return fn_flatten(
                curr_input,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=False,
            )

        def linear():
            return fn_linear(
                curr_input,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=False,
            )

        def pool():
            return fn_pool(
                curr_input,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=False,
            )

        def relu():
            return fn_relu(
                curr_input,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=False,
            )

        def softmax():
            return fn_softmax(
                curr_input,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=False,
            )

        switcher = {
            "conv": conv,
            "flatten": flatten,
            "linear": linear,
            "pool": pool,
            "relu": relu,
            "softmax": softmax,
        }

        def switch(type):
            return switcher.get(type)()

        activations[i], _, _ = switch(layers[i]["type"])
        curr_input = activations[i]

    output = activations[-1]
    return output, activations
