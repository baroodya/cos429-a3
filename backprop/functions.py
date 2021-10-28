import math


"""
Defines forward and backward passes through different computational graphs.

Students should complete the implementation of all functions in this file.
"""


def f1(x1, w1, x2, w2, b, y):
    """
    Computes the forward and backward pass through the computational graph f1
    from the homework PDF.

    A few clarifications about the graph:
    - The subtraction node in the graph computes d = y_hat - y
    - The ^2 node squares its input

    Inputs:
    - x1, w1, x2, w2, b, y: Python floats

    Returns a tuple of:
    - L: Python scalar giving the output of the graph
    - grads: A tuple (grad_x1, grad_w1, grad_x2, grad_w2, grad_b, grad_y)
    giving the derivative of the output L with respect to each input.
    """
    # Forward pass: compute loss
    L = None

    a1 = x1 * w1
    a2 = x2 * w2
    y_hat = a1 + a2 + b
    d = y_hat - y
    L = d ** 2

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    # Backward pass: compute gradients
    grad_x1, grad_w1, grad_x2, grad_w2 = None, None, None, None
    grad_b, grad_y = None, None

    grad_d = 2 * d
    grad_y = grad_d * -1
    grad_b = grad_d * 1

    grad_x1 = grad_b * w1
    grad_w1 = grad_b * x1

    grad_x2 = grad_b * w2
    grad_w2 = grad_b * x2

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    grads = (grad_x1, grad_w1, grad_x2, grad_w2, grad_b, grad_y)
    return L, grads


def f2(x):
    """
    Computes the forward and backward pass through the computational graph f2
    from the homework PDF.

    A few clarifications about this graph:
    - The "x2" node multiplies its input by the constant 2
    - The "+1" and "-1" nodes add or subtract the constant 1
    - The division node computes y = t / b

    Inputs:
    - x: Python float

    Returns a tuple of:
    - y: Python float
    - grads: A tuple (grad_x,) giving the derivative of the output y with
      respect to the input x
    """
    # Forward pass: Compute output
    y = None

    d = 2 * x
    e = math.exp(d)
    t = e - 1
    b = e + 1
    y = t / b

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    # Backward pass: Compute gradients
    grad_x = None

    grad_t = 1 / b
    grad_b = -t / (b ** 2)

    grad_e = grad_t * 1 + grad_b * 1
    grad_d = math.exp(d) * grad_e
    grad_x = 2 * grad_d

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y, (grad_x,)


def f3(s1, s2, y):
    """
    Computes the forward and backward pass through the computational graph f3
    from the homework PDF.

    A few clarifications about the graph:
    - The input y is an integer with y == 1 or y == 2; you do not need to
      compute a gradient for this input.
    - The division nodes compute p1 = e1 / d and p2 = e2 / d
    - The choose(p1, p2, y) node returns p1 if y is 1, or p2 if y is 2.

    Inputs:
    - s1, s2: Python floats
    - y: Python integer, either equal to 1 or 2

    Returns a tuple of:
    - L: Python scalar giving the output of the graph
    - grads: A tuple (grad_s1, grad_s2) giving the derivative of the output L
    with respect to the inputs s1 and s2.
    """
    assert y == 1 or y == 2
    # Forward pass: Compute loss
    L = None

    e1 = math.exp(s1)
    e2 = math.exp(s2)
    d = e1 + e2
    p1 = e1 / d
    p2 = e2 / d

    p_plus = None
    if y == 1:
        p_plus = p1
    else:
        p_plus = p2

    L = -(math.log(p_plus))

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    # Backward pass: Compute gradients
    grad_s1, grad_s2 = None, None

    grad_p_plus = -(1 / p_plus)
    if y == 1:
        grad_p1 = grad_p_plus * 1
        grad_p2 = grad_p_plus * 0
        this_e = e1
    else:
        grad_p1 = grad_p_plus * 0
        grad_p2 = grad_p_plus * 1
        this_e = e2

    grad_d = grad_p1 * (-this_e / (d ** 2))
    grad_d += grad_p2 * (-this_e / (d ** 2))

    grad_e1 = grad_p1 * (1 / d)
    grad_e2 = grad_p2 * (1 / d)

    grad_e1 += grad_d * 1
    grad_e2 += grad_d * 1

    grad_s1 = grad_e1 * math.exp(s1)
    grad_s2 = grad_e2 * math.exp(s2)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    grads = (grad_s1, grad_s2)
    return L, grads


def f3_y1(s1, s2):
    """
    Helper function to compute f3 in the case where y = 1

    Inputs:
    - s1, s2: Same as f3

    Outputs: Same as f3
    """
    return f3(s1, s2, y=1)


def f3_y2(s1, s2):
    """
    Helper function to compute f3 in the case where y = 2

    Inputs:
    - s1, s2: Same as f3

    Outputs: Same as f3
    """
    return f3(s1, s2, y=2)
