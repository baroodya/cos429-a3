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
    vel = hyper_params["velocity"]
    rho = hyper_params["rho"]

    updated_model = model

    # TODO: Update the weights of each layer in your model based on the calculated gradients
    for i in range(num_layers):
        curr = updated_model["layers"][i]["params"]

        vel[i]["W"] = rho * vel[i]["W"] + grads[i]["W"]
        vel[i]["b"] = rho * vel[i]["b"] + grads[i]["b"]

        updated_model["layers"][i]["params"]["W"] = curr["W"] - (
            vel[i]["W"] * (a + lmd)
        )
        updated_model["layers"][i]["params"]["b"] = curr["b"] - (
            vel[i]["b"] * a
        )

        # updated_model["layers"][i]["params"]["W"] = curr["W"] - (
        #     grads[i]["W"] * a * lmd
        # )

        # updated_model["layers"][i]["params"]["b"] = curr["b"] - (
        #     grads[i]["b"] * a
        # )

    return updated_model, vel
