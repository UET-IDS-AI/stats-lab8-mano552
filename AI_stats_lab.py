import numpy as np


# -------------------------------------------------
# Question 1: Continuous pair on the unit square
# -------------------------------------------------

def joint_cdf_unit_square(x, y):
    """
    Return the joint CDF F_XY(x, y) for (X, Y) uniform on the unit square.

    F_XY(x, y) =
        0                   if x <= 0 or y <= 0
        x*y                 if 0 < x < 1 and 0 < y < 1
        x                   if 0 < x < 1 and y >= 1
        y                   if x >= 1 and 0 < y < 1
        1                   if x >= 1 and y >= 1
    """
    if x <= 0 or y <= 0:
        return 0
    elif 0 < x < 1 and 0 < y < 1:
        return x * y
    elif 0 < x < 1 and y >= 1:
        return x
    elif x >= 1 and 0 < y < 1:
        return y
    elif x >= 1 and y >= 1:
        return 1
    return 0


def rectangle_probability(x1, x2, y1, y2):
    """
    Compute P(x1 < X <= x2, y1 < Y <= y2)
    using the joint CDF rectangle formula.
    """
    return (
        joint_cdf_unit_square(x2, y2)
        - joint_cdf_unit_square(x1, y2)
        - joint_cdf_unit_square(x2, y1)
        + joint_cdf_unit_square(x1, y1)
    )


def marginal_fx_unit_square(x):
    """
    Return the marginal PDF f_X(x)
    """
    if 0 < x < 1:
        return 1
    return 0


def marginal_fy_unit_square(y):
    """
    Return the marginal PDF f_Y(y)
    """
    if 0 < y < 1:
        return 1
    return 0


# -------------------------------------------------
# Question 2: Joint PMF, marginals, independence
# -------------------------------------------------

def joint_pmf_heads(x, y):
    """
    Return P_XY(x, y)
    """
    table = {
        (0, 0): 1/4,
        (0, 1): 1/4,
        (0, 2): 0,
        (1, 0): 0,
        (1, 1): 1/4,
        (1, 2): 1/4
    }
    return table.get((x, y), 0)


def marginal_px_heads(x):
    """
    Return P_X(x) by summing over y
    """
    return sum(joint_pmf_heads(x, y) for y in [0, 1, 2])


def marginal_py_heads(y):
    """
    Return P_Y(y) by summing over x
    """
    return sum(joint_pmf_heads(x, y) for x in [0, 1])


def check_independence_heads():
    """
    Return True if X and Y are independent, else False.
    """
    for x in [0, 1]:
        for y in [0, 1, 2]:
            if joint_pmf_heads(x, y) != marginal_px_heads(x) * marginal_py_heads(y):
                return False
    return True
