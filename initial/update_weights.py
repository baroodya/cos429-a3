import numpy as np


def update_weights(model, grads, hyper_params):
    """
    Update the weights of each layer in your model based on the calculated gradients
    Args:
        model: Dictionary holding the model
        grads: A list of gradients of each layer in model["layers"]
        hyper_params:
            hyper_params['learning_rate']
            hyper_params['weight_decay']: Should be applied to W only.
    Returns:
        updated_model:  Dictionary holding the updated model
    """
    num_layers = len(grads)
    a = hyper_params["learning_rate"]
    lmd = hyper_params["weight_decay"]
    updated_model = model

    # TODO: Update the weights of each layer in your model based on the calculated gradients
    for i in range(num_layers):
        curr = updated_model["layers"][i]["params"]

        norm = np.linalg.norm(curr["W"])

        updated_model["layers"][i]["params"]["W"] = (
            curr["W"] - (grads[i]["W"] * a) + 2 * curr["W"] * lmd
        )
        updated_model["layers"][i]["params"]["b"] = curr["b"] - (
            grads[i]["b"] * a
        )

    return updated_model
