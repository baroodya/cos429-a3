from layers.fn_conv import fn_conv
from pyc_code.fn_conv_ import fn_conv as fn_conv_

from inference import inference
from pyc_code.inference_ import inference as inference_

from pyc_code.loss_crossentropy_ import loss_crossentropy

from calc_gradient import calc_gradient
from pyc_code.calc_gradient_ import calc_gradient as calc_gradient_

from update_weights import update_weights
from pyc_code.update_weights_ import (
    update_weights as ref_update_weights,
)


def test_implementations(model, ref_model, X_train, y_train, params):
    batch_size = params.get("batch_size", 128)
    start = 0
    end = start + batch_size
    batch = X_train[:, :, :, start:end]
    assert batch.shape[:-1] == X_train.shape[:-1]
    assert batch.shape[-1] == batch_size
    for i in range(len(model["layers"])):
        assert (
            model["layers"][i]["params"]["W"]
            == ref_model["layers"][i]["params"]["W"]
        ).all()

    test_fn_conv_forward(model, ref_model, batch)
    output, activations = test_inference(model, batch)

    _, dv_output = loss_crossentropy(
        output, y_train[start:end], {}, backprop=True
    )

    grads = test_calc_gradient(model, batch, activations, dv_output)

    # Learning rate
    lr = params.get("learning_rate", 0.01)
    # Weight decay
    wd = params.get("weight_decay", 0.0005)

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr, "weight_decay": wd}

    _ = test_update_weights(model, ref_model, grads, update_params)


def test_fn_conv_forward(model, ref_model, batch):
    my_output, _, _ = fn_conv(
        batch,
        model["layers"][0]["params"],
        model["layers"][0]["hyper_params"],
        backprop=False,
    )

    ref_output, _, _ = fn_conv_(
        batch,
        ref_model["layers"][0]["params"],
        ref_model["layers"][0]["hyper_params"],
        backprop=False,
    )

    if (my_output != ref_output).all():
        print("My output:\n", my_output[0])
        print("Reference output:\n", ref_output[0])

        print("Difference:\n", (my_output - ref_output))
        exit(1)
    else:
        print("fn_conv forward pass is correct!")


def test_fn_conv_backward(model, ref_model, batch, dv_output):
    _, my_dv_input, my_grads = fn_conv(
        batch,
        model["layers"][0]["params"],
        model["layers"][0]["hyper_params"],
        backprop=True,
        dv_output=dv_output,
    )

    _, ref_dv_input, ref_grads = fn_conv(
        batch,
        ref_model["layers"][0]["params"],
        ref_model["layers"][0]["hyper_params"],
        backprop=True,
        dv_output=dv_output,
    )

    if ref_dv_input != my_dv_input:
        print("My Gradients:\n", my_dv_input)
        print("Reference Gradients:\n", ref_dv_input)
        return ref_grads
    return my_grads


def test_calc_gradient(model, batch, activations, dv_output):

    ref_grads = calc_gradient_(model, batch, activations, dv_output)
    my_grads = calc_gradient(model, batch, activations, dv_output)

    correct = True
    for i in range(len(my_grads)):
        if (my_grads[i]["W"] != ref_grads[i]["W"]).any() and (
            my_grads[i]["b"] != ref_grads[i]["b"]
        ).any():
            print("Layer", i)
            print(
                "My output:\n", my_grads[i]["W"], "\n", my_grads[i]["b"]
            )
            print(
                "Reference output:\n",
                ref_grads[i]["W"],
                "\n",
                ref_grads[i]["b"],
            )
            correct = False

    if not correct:
        return ref_grads
    else:
        print("fn_conv backward pass is correct!")
        return my_grads


def test_update_weights(model, ref_model, grads, hyper_params):
    my_updated_model = update_weights(model, grads, hyper_params)
    ref_updated_model = ref_update_weights(
        ref_model, grads, hyper_params
    )

    correct = True

    for i in range(len(my_updated_model["layers"])):
        if (
            abs(
                my_updated_model["layers"][i]["params"]["W"]
                - ref_updated_model["layers"][i]["params"]["W"]
            )
            > 0.0001
        ).any():
            print(
                "Model differences at layer:",
                i,
                "\n",
                my_updated_model["layers"][i]["params"]["W"]
                - ref_updated_model["layers"][i]["params"]["W"],
            )
            correct = False

    if correct:
        print("update_weights is correct!")
        return my_updated_model
    else:
        return ref_updated_model


def test_inference(model, batch):
    my_output, my_activations = inference(model, batch)
    ref_output, ref_activations = inference_(model, batch)

    correct = True

    if ((my_output - ref_output) > 0.000001).all():
        print("My output:\n", my_output)
        print("Reference output:\n", ref_output)
        correct = False
    #  if (my_output != ref_output).all():
    #      print("My output:\n", my_output)
    #      print("Reference output:\n", ref_output)
    #      correct = False

    if correct:
        print("Inference is correct!")
        return my_output, my_activations
    else:
        return ref_output, ref_activations
