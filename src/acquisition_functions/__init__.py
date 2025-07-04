from .random import random
from .desirability import desirability, desirability_wAD
from .uncertainty import uncertainty
# from .ucb import ucb
from .kmeans import kmeans
from .kmedoids import kmedoids
from .harmonic import harmonic
from .coverage import coverage
from .desired_coverage import desired_coverage
from .greediverse import greediverse
from .similarity import similarity
from .best_similarity import best_similarity
from .dissimilarity import dissimilarity
from .best_dissimilarity import best_dissimilarity

from .ratio import ratio
from .topsis import topsis
from .gra import gra

from .oracle import oracle
from .desired_diversity import desired_diversity

from .spread import spread
from .desired_spread import desired_spread

def get_strategies_dict():
    return {
        'random': random,
        'desirability': desirability,
        'uncertainty': uncertainty,
        'kmeans': kmeans,
        'kmedoids': kmedoids,
        'harmonic': harmonic,
        'coverage': coverage,
        'desired_coverage': desired_coverage,
        'greediverse': greediverse,
        'similarity': similarity,
        'best_similarity': best_similarity,
        'dissimilarity': dissimilarity,
        'best_dissimilarity': best_dissimilarity,
        'gra': gra,
        'topsis': topsis,
        'ratio': ratio,
        'oracle': oracle,
        'desired_diversity': desired_diversity,
        'desirability_wAD': desirability_wAD,
        'spread': spread,
        'desired_spread': desired_spread

    }
