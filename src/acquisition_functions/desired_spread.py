import numpy as np
import pandas as pd

def desired_spread(self, batch_size):

    if len(self.smiles_pool) > batch_size:
        smiles_pool = self.smiles_pool
        pool_scores = self.pool_df[f'{self.scoring}_predicted_score'].values
        selectable_df = pd.DataFrame()
        selectable_df['SMILES'] = smiles_pool
        selectable_df['score'] = pool_scores

        smiles_train = list(self.smiles_train)

        smiles_selected = []

        for i in range(batch_size):

            tmp_df = selectable_df.copy()
            tmp_df = tmp_df[~tmp_df['SMILES'].isin(smiles_selected)]

            desired_spread_scores = []
            for index, row in tmp_df.iterrows():
                smiles = row['SMILES']
                score = row['score']
                max_similarity = self.similarity_matrix.loc[smiles, smiles_train + smiles_selected].max()
                nearest_neighbor_distance = 1 - max_similarity
                desired_spread_score = score * nearest_neighbor_distance
                desired_spread_scores.append(desired_spread_score)

            tmp_df['desired_spread_score'] = np.vstack(desired_spread_scores)

            selected_molecule = tmp_df.loc[tmp_df['desired_spread_score'].idxmax()]['SMILES']
            smiles_selected.append(selected_molecule)
    else:
        smiles_selected = self.smiles_pool
    return smiles_selected