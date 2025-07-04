import numpy as np
import pandas as pd

# Array functions
def multi_argmax(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Return the indices of the n largest values in the array.
    
    Parameters:
    arr (np.ndarray): The input array.
    n (int): The number of largest values to return.
    
    Returns:
    np.ndarray: Indices of the n largest values.
    """
    return np.argsort(arr)[-n:]

def multi_argmin(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Return the indices of the n smallest values in the array.
    
    Parameters:
    arr (np.ndarray): The input array.
    n (int): The number of smallest values to return.
    
    Returns:
    np.ndarray: Indices of the n smallest values.
    """
    return np.argsort(arr)[:n]

def sum_values(row, value_dict):
    total = 0
    for key, value in value_dict.items():
        # Check if the key exists in the row and the value is not NaN
        if key in row and not pd.isna(row[key]):
            total += value
    return total