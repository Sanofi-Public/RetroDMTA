import numpy as np
import pandas as pd

def spread(self, batch_size):

    if len(self.smiles_pool) > batch_size:
        smiles_pool = self.smiles_pool
        selectable_df = pd.DataFrame()
        selectable_df['SMILES'] = smiles_pool

        smiles_train = list(self.smiles_train)

        smiles_selected = []

        for i in range(batch_size):

            tmp_df = selectable_df.copy()
            tmp_df = tmp_df[~tmp_df['SMILES'].isin(smiles_selected)]

            nearest_neighbor_distances = []
            for index, row in tmp_df.iterrows():
                smiles = row['SMILES']
                max_similarity = self.similarity_matrix.loc[smiles, smiles_train + smiles_selected].max()
                nearest_neighbor_distance = 1 - max_similarity
                nearest_neighbor_distances.append(nearest_neighbor_distance)
                
            tmp_df['distance'] = np.vstack(nearest_neighbor_distances)

            selected_molecule = tmp_df.loc[tmp_df['distance'].idxmax()]['SMILES']
            smiles_selected.append(selected_molecule)
    else:
        smiles_selected = self.smiles_pool
    return smiles_selected