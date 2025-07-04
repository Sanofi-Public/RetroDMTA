import numpy as np

def sigmoid_HTB(x, LTV, HTV, q=1, HS=0.05, LS=0.05, cutoff=0.05):
    """
    Higher-the-better sigmoid function.
    
    Parameters:
        x (float): The input value.
        LTV (float): Lower asymptotic value.
        HTV (float): Higher asymptotic value.
        q (float): Quantile for determining the sigmoid center, default is 1.
        HS (float): High saturation level, default is 0.05.
        LS (float): Low saturation level, default is 0.05.
        cutoff (float): Minimum value to return, default is 0.05.

    Returns:
        float: Sigmoid function result for HTB.
    """
    quantile_value = np.quantile([LTV, HTV], q=1-q)

    if LTV == HTV:
        if x > HTV:
            return 1
        else:
            return cutoff
    else:
        value = (x - quantile_value) * 10 / (LTV - HTV)
        clipped_value = np.clip(value, -700, 700)
        exp_value = np.exp(clipped_value)
        sigmoid_value = (1 / (1 + exp_value)) * (1 - HS) + LS
    return np.round(sigmoid_value, 4)

def sigmoid_LTB(x, LTV, HTV, q=1, HS=0.05, LS=0.05, cutoff=0.05):
    """
    Lower-the-better sigmoid function.
    
    Parameters:
        x (float): The input value.
        LTV (float): Lower asymptotic value.
        HTV (float): Higher asymptotic value.
        q (float): Quantile for determining the sigmoid center, default is 1.
        HS (float): High saturation level, default is 0.05.
        LS (float): Low saturation level, default is 0.05.
        cutoff (float): Minimum value to return, default is 0.05.

    Returns:
        float: Sigmoid function result for LTB.
    """
    quantile_value = np.quantile([LTV, HTV], q=q)

    if LTV == HTV:
        if x < LTV:
            return 1
        else:
            return cutoff
    else:
        value = (x - quantile_value) * 10 / (HTV - LTV)
        clipped_value = np.clip(value, -700, 700)
        exp_value = np.exp(clipped_value)
        sigmoid_value = (1 / (1 + exp_value)) * (1 - HS) + LS
    return np.round(sigmoid_value, 4)

def sigmoid_INT(x, LTV, HTV, q=1, HS=0.05, LS=0.05, cutoff=0.05):
    """
    Interval sigmoid function combining HTB and LTB.
    
    Parameters:
        x (float): The input value.
        LTV (float): Lower asymptotic value.
        HTV (float): Higher asymptotic value.
        q (float): Quantile for determining the sigmoid center, default is 1.
        HS (float): High saturation level, default is 0.05.
        LS (float): Low saturation level, default is 0.05.
        cutoff (float): Minimum value to return, default is 0.05.

    Returns:
        float: Sigmoid function result for intermediate values.
    """
    mean_value = (LTV + HTV) / 2
    if x < mean_value:
        return sigmoid_HTB(x, LTV, mean_value, q, HS, LS, cutoff)
    else:
        return sigmoid_LTB(x, mean_value, HTV, q, HS, LS, cutoff)

# Linear functions
def linear_LTB(x, LTV, HTV):
    value = (x - HTV) / (LTV - HTV)
    if value < 0.05:
        return 0.05
    elif value > 1:
        return 1
    else:
        return value

def linear_HTB(x, LTV, HTV):
    value = (x - LTV) / (HTV - LTV)
    if value < 0.05:
        return 0.05
    elif value > 1:
        return 1
    else:
        return value

def linear_INT(x, LTV, HTV, divider=25):
    factor = (HTV + LTV)/divider
    mean = (HTV+LTV)/2
    HTB_HTV = LTV + factor
    HTB_LTV = LTV

    LTB_LTV = HTV - factor
    LTB_HTV = HTV

    if x < mean:
        return linear_HTB(x, HTB_HTV, HTB_LTV)
    else:
        return linear_LTB(x, LTB_HTV, LTB_LTV)
