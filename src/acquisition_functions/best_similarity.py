import numpy as np
import pandas as pd

def best_similarity(self, batch_size):

    if len(self.smiles_pool) > batch_size:
        if len(self.good_in_train) != 0:
            reduced_similarity_matrix = self.similarity_matrix.loc[self.good_in_train, self.smiles_pool]
            mean_matrix = reduced_similarity_matrix.mean(axis=0).sort_values(ascending=False)
            smiles_selected = list(mean_matrix.index[:batch_size])
        else:
            reduced_similarity_matrix = self.similarity_matrix.loc[self.smiles_train, self.smiles_pool]
            mean_matrix = reduced_similarity_matrix.mean(axis=0).sort_values(ascending=False)
            smiles_selected = list(mean_matrix.index[:batch_size])
    else:
        smiles_selected = self.smiles_pool
    return smiles_selected