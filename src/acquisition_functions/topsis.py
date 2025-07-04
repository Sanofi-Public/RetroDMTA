import numpy as np
from utils import multi_argmax

# Function: TOPSIS
def topsis_method(data, weights):
    v_ij = data * weights
    p_ideal_A = np.ones(data.shape[1]) * weights
    n_ideal_A = np.ones(data.shape[1]) * 0.05 * weights

    p_s_ij = (v_ij - p_ideal_A)**2
    p_s_ij = np.sum(p_s_ij, axis = 1)**(1/2)
    n_s_ij = (v_ij - n_ideal_A)**2
    n_s_ij = np.sum(n_s_ij, axis = 1)**(1/2)
    c_i    = n_s_ij / ( p_s_ij  + n_s_ij )
    return c_i


def topsis(self, batch_size):
    if len(self.pool_df) > batch_size:

        assert len(self.smiles_pool) == len(self.pool_df)

        tmp_df = self.pool_df.filter(regex='^(normalized_predicted)').copy()
        tmp_df.columns = tmp_df.columns.str.replace('normalized_predicted_', '', regex=False)

        weights = []
        properties = []

        for prop in tmp_df.columns:
            weights.append(self.blueprint.loc[self.blueprint['PROPERTIES'] == prop, 'WEIGHT'].values[0])
            properties.append(prop)
        
        weights = np.array(weights)
        weights = weights / sum(weights)
        
        dataset = tmp_df.values
        topsis_scores = topsis_method(dataset, weights)
        selected_smiles = self.pool_df.iloc[multi_argmax(topsis_scores, batch_size)]['SMILES'].values
    else:
        selected_smiles = self.pool_df['SMILES'].values
    return selected_smiles
