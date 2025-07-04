import numpy as np
from utils import multi_argmax

def gra_method(dataset, weights, epsilon = 0.5):
    deviation_sequence = 1 - dataset
    gra_coefficient = epsilon/(deviation_sequence + epsilon)
    gra_grade = np.sum(gra_coefficient*weights, axis = 1) / dataset.shape[0]
    return gra_grade 


def gra(self, batch_size):
    if len(self.pool_df) > batch_size:

        assert len(self.smiles_pool) == len(self.pool_df)

        tmp_df = self.pool_df.filter(regex='^(normalized_predicted)').copy()
        tmp_df.columns = tmp_df.columns.str.replace('normalized_predicted_', '', regex=False)

        weights = []
        properties = []

        for prop in tmp_df.columns:
            weights.append(self.blueprint.loc[self.blueprint['PROPERTIES'] == prop, 'WEIGHT'].values[0])
            properties.append(prop)
            
        dataset = tmp_df.values
        gra_grades = gra_method(dataset, weights)
        selected_smiles = self.pool_df.iloc[multi_argmax(gra_grades, batch_size)]['SMILES'].values
    else:
        selected_smiles = self.pool_df['SMILES'].values
    return selected_smiles