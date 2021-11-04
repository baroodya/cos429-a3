import sys
from datetime import datetime

sys.path += ["layers"]
import numpy as np
from layers.loss_crossentropy import loss_crossentropy

######################################################
# Set use_pcode to True to use the provided pyc code
# for inference, calc_gradient, loss_crossentropy and update_weights
use_pcode = True

# You can modify the imports of this section to indicate
# whether to use the provided pyc or your own code for each of the four functions.
if use_pcode:
    # import the provided pyc implementation
    sys.path += ["pyc_code"]
    from pyc_code.inference_ import inference
    from pyc_code.calc_gradient_ import calc_gradient
    from pyc_code.update_weights_ import update_weights
else:
    # import your own implementation
    from inference import inference
    from calc_gradient import calc_gradient
    from update_weights import update_weights

######################################################

# Calculate the accuracy of a specific output. Assumes soft max final layer.
def get_accuracy(output, labels, num_inputs):
    predictions = np.argmax(output, axis=0)

    _labels = labels.astype("int32")
    correct = (predictions - _labels) == 0

    accuracy = np.sum(correct) / num_inputs
    return accuracy


def train(model, input, label, params, numIters):
    """
    This training function is written specifically for classification,
    since it uses crossentropy loss and tests accuracy assuming the final output
    layer is a softmax layer. These can be changed for more general use.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [num_inputs]
        label: [num_inputs]
        params: Paramters for configuring training
            params["learning_rate"]
            params["weight_decay"]
            params["batch_size"]
            params["save_file"]
            params["X_test"]
            params["y_test"]
            Free to add more parameters to this dictionary for your convenience of training.
        numIters: Number of training iterations
    """
    start_time = datetime.now()
    # Initialize training parameters
    # Learning rate
    lr = params.get("learning_rate", 0.01)
    # Weight decay
    wd = params.get("weight_decay", 0.0005)
    # velovity for momentum
    vel = params.get("velocity", 0)
    # friction for momentum
    rho = params.get("friction", 0.99)
    # Batch size
    batch_size = params.get("batch_size", 128)
    # There is a good chance you will want to save your network model during/after
    # training. It is up to you where you save and how often you choose to back up
    # your model. By default the code saves the model in 'model.npz'.
    save_file = params.get("save_file", "model.npz")

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {
        "learning_rate": lr,
        "weight_decay": wd,
        "velocity": vel,
        "rho": rho,
    }

    X_test = params["X_test"]
    y_test = params["y_test"]

    training_accuracies = []
    testing_accuracies = []

    num_training_inputs = input.shape[-1]
    num_test_inputs = X_test.shape[-1]
    training_loss = np.zeros((numIters,))
    testing_loss = np.zeros((int(numIters / 100),))

    for i in range(numIters):
        #   (1) Select a subset of the input to use as a batch
        start = (i * batch_size) % (num_training_inputs - batch_size)
        end = start + batch_size
        batch = input[:, :, :, start:end]
        assert batch.shape[:-1] == input.shape[:-1]
        assert batch.shape[-1] == batch_size

        #   (2) Run inference on the batch
        output, activations = inference(model, batch)

        #   (3) Calculate loss and determine accuracy
        training_loss[i], dv_output = loss_crossentropy(
            output, label[start:end], {}, backprop=True
        )
        training_accuracies.append(
            get_accuracy(output, label[start:end], batch_size)
        )

        # break if loss has plateaued
        if np.isnan(training_loss[i]):
            break

        # Check the testing/validation data every 50 iterations
        if i % 50 == 0:
            start = (i * batch_size) % (num_test_inputs - batch_size)
            end = start + batch_size
            batch = X_test[:, :, :, start:end]
            assert batch.shape[:-1] == X_test.shape[:-1]
            assert batch.shape[-1] == batch_size
            #   (2) Run inference on the batch
            output, activations = inference(model, batch)
            #   (3) Calculate loss and determine accuracy
            testing_loss[i], _ = loss_crossentropy(
                output, y_test[start:end], {}, backprop=False
            )
            testing_accuracies.append(
                get_accuracy(output, y_test[start:end], batch_size)
            )

        #   (4) Calculate gradients
        grads = calc_gradient(
            model,
            batch,
            activations,
            dv_output,
        )

        if i == 0:
            for j in range(len(grads)):
                shapes = {
                    "W": np.zeros(grads[j]["W"].shape),
                    "b": np.zeros(grads[j]["b"].shape),
                }
                update_params["velocity"].append(shapes)

        #   (5) Update the weights of the model
        model = update_weights(model, grads, update_params)
        # update_params["velocity"] = vel

        # Optionally,
        #   (1) Monitor the progress of training
        if i % 10 == 0:
            percent_finished = np.round(((i / numIters) * 100), 3)
            print(
                percent_finished,
                "% done. ",
                "Loss = ",
                np.round(training_loss[i], 3),
                ". Training Accuracy = ",
                np.round(training_accuracies[-1], 3),
                ". Testing Accuracy = ",
                np.round(testing_accuracies[-1], 3),
                sep="",
            )

        #   (2) Save your learnt model, using ``np.savez(save_file, **model)``
        if i % 50 == 0:
            np.savez(save_file, **model)

    print("Training model... 100% done.")

    print("Calculation final testing accuracy...", end="\r")
    output, _ = inference(model, X_test)
    testing_accuracies.append(
        get_accuracy(output, y_test, num_test_inputs)
    )

    duration = datetime.now() - start_time

    return (
        duration,
        training_loss,
        testing_loss,
        training_accuracies,
        testing_accuracies,
    )
