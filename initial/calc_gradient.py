import numpy as np

import sys

sys.path += ["layers"]
sys.path += ["pyc_code"]

from fn_conv import fn_conv
from fn_flatten import fn_flatten
from pyc_code.fn_linear_ import fn_linear
from fn_pool import fn_pool
from fn_relu import fn_relu
from pyc_code.fn_softmax_ import fn_softmax


def calc_gradient(model, input, layer_acts, dv_output):
    """
    Calculate the gradient at each layer, to do this you need dv_output
    determined by your loss function and the activations of each layer.
    The loop of this function will look very similar to the code from
    inference, just looping in reverse.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
        layer_acts: A list of activations of each layer in model["layers"]
        dv_output: The partial derivative of the loss with respect to each element in the output matrix of the last layer.
    Returns:
        grads:  A list of gradients of each layer in model["layers"]
    """
    num_layers = len(model["layers"])
    grads = [
        None,
    ] * num_layers

    # TODO: Determine the gradient at each layer.
    #       Remember that back-propagation traverses
    #       the model in the reverse order.
    layers = model["layers"]

    new_dv_output = dv_output
    for i in np.flip(range(num_layers)):
        if i > 0:
            activations = layer_acts[i - 1]
        else:
            activations = input

        def conv():
            # print("Conv activations:", activations)
            # print("Conv dv_output:", new_dv_output)
            grads = fn_conv(
                activations,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=True,
                dv_output=new_dv_output,
            )
            # print("Conv gradients:", grads)
            return grads

        def flatten():
            return fn_flatten(
                activations,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=True,
                dv_output=new_dv_output,
            )

        def linear():
            return fn_linear(
                activations,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=True,
                dv_output=new_dv_output,
            )

        def pool():
            return fn_pool(
                activations,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=True,
                dv_output=new_dv_output,
            )

        def relu():
            return fn_relu(
                activations,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=True,
                dv_output=new_dv_output,
            )

        def softmax():
            return fn_softmax(
                activations,
                layers[i]["params"],
                layers[i]["hyper_params"],
                backprop=True,
                dv_output=new_dv_output,
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

        _, new_dv_output, grads[i] = switch(layers[i]["type"])

    return grads
