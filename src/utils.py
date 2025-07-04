# Utils
import datetime

# Data
import numpy as np
import pandas as pd

# Cheminformatics
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


def multi_argmax(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Returns the indices of the top `n` maximum values in the array.

    Parameters:
    - arr (np.ndarray): The input array to process.
    - n (int): The number of indices corresponding to the largest values to return.

    Returns:
    - np.ndarray: An array of indices of the top `n` maximum values in descending order.
    """
    return np.argsort(arr)[-n:]


def multi_argmin(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Returns the indices of the top `n` minimum values in the array.

    Parameters:
    - arr (np.ndarray): The input array to process.
    - n (int): The number of indices corresponding to the smallest values to return.

    Returns:
    - np.ndarray: An array of indices of the top `n` minimum values in ascending order.
    """
    return np.argsort(arr)[:n]


def convert_to_BitVect(array):
    """
    Converts a binary array into an RDKit ExplicitBitVect object.

    Parameters:
    - array (list or np.ndarray): A binary array (list or numpy array) where each element is 0 or 1.

    Returns:
    - ExplicitBitVect: An RDKit ExplicitBitVect object with bits set according to the input array.
    """
    # Initialize an ExplicitBitVect with a size equal to the length of the input array
    bitvect = ExplicitBitVect(len(array))
    
    # Set bits in the ExplicitBitVect based on the values in the input array
    for idx, val in enumerate(array):
        if val:
            bitvect.SetBit(idx)
    
    return bitvect

def compute_similarity_matrix(smiles, fingerprints):

    fingerprints = [convert_to_BitVect(fp) for fp in fingerprints]
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        if fingerprints[i] is not None:
            similarities = BulkTanimotoSimilarity(fingerprints[i], fingerprints[i:])
            similarity_matrix[i, i:] = similarities
            similarity_matrix[i:, i] = similarities

    similarity_df = pd.DataFrame(similarity_matrix, index=smiles, columns=smiles)

    return similarity_df

# Used to compute the experimental scores
def sum_values(row, my_dict):
    total = 0
    for key, value in my_dict.items():
        if not pd.isna(row[key]):
            total += value
    return total

def get_time():
    return (datetime.datetime.now()+datetime.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")


def get_bs_and_strat(batch_size, strategy_type, strategy, iteration, batchsize_per_iteration):

    if 0 < batch_size < 1:

        if iteration not in batchsize_per_iteration.keys():
            batch_size = 0
        elif batchsize_per_iteration[iteration] == 0:
            batch_size = 0
        else:
            batch_size = max(1, round(batch_size * batchsize_per_iteration[iteration]))
            
    batch_sizes = [batch_size]
    selection_strategies = [strategy]
    
    # if strategy_type == 'pure':
    #     batch_sizes = [batch_size]
    #     selection_strategies = [strategy]
    # else:
    #     nb_greedy = int(strategy_type.split('_')[0])
    #     nb_other = int(strategy_type.split('_')[1])
    #     assert nb_greedy + nb_other == batch_size, "Number selected using desirability and other samples should sum to batch_size"
    #     selection_strategies = ['desirability', strategy]
    #     batch_sizes = [nb_greedy, nb_other] 
    return selection_strategies, batch_sizes

def process_strategy_name(strategy):

    if 'greediverse' in strategy:

        strategy_name = 'greediverse'

        parameters = strategy.split('_')[-2:]
        l = parameters[0]
        t = parameters[1]

        if l[0] == '0':
            greediverse_lambda = float(l[0] + '.' + l[1:])
        else:
            greediverse_lambda = float(l)

        if t[0] == '0':
            greediverse_threshold = float(t[0] + '.' + t[1:])
        else:
            greediverse_threshold = float(t)

        return strategy_name, greediverse_lambda, greediverse_threshold, None, None

    elif 'ratio' in strategy:

        strategy_name = 'ratio'

        parameters = strategy.split('_')[-1]

        if parameters[0] == '0':
            ratio_epsilon = float(parameters[0] + '.' + parameters[1:])
        else:
            ratio_epsilon = float(parameters)
        
        return strategy_name, None, None, ratio_epsilon, None
    
    elif 'desired_diversity' in strategy:

        strategy_name = 'desired_diversity'

        parameters = strategy.split('_')[-1]

        if parameters[0] == '0':
            desired_diversity_threshold = float(parameters[0] + '.' + parameters[1:])
        else:
            desired_diversity_threshold = float(parameters)
        
        return strategy_name, None, None, None, desired_diversity_threshold

    else:
        strategy_name = strategy
        return strategy_name, None, None, None, None