import pandas as pd
import numpy as np

def desirability(self, batch_size):
    if len(self.pool_df) > batch_size:
        selected_smiles = self.pool_df.sort_values(by=f'{self.scoring}_predicted_score', ascending=False)['SMILES'].values[:batch_size]
    else:
        selected_smiles = self.pool_df['SMILES'].values
    return selected_smiles

def desirability_wAD(self, batch_size):

    if len(self.smiles_pool) > batch_size:
        smiles_pool = self.smiles_pool
        pool_scores = self.pool_df[f'{self.scoring}_predicted_score'].values
        selectable_df = pd.DataFrame()
        selectable_df['SMILES'] = smiles_pool
        selectable_df['score'] = pool_scores

        ad_validity = []
        for index, row in selectable_df.iterrows():
            smiles = row['SMILES']
            similarity = self.similarity_matrix.loc[smiles, self.smiles_train].max()

            if similarity > 0.7:
                ad_validity.append(1)
            else:
                ad_validity.append(0)

        selectable_df['ad_validity'] = np.vstack(ad_validity)
        selectable_df.sort_values(by=['ad_validity', 'score'], ascending=[False, False], inplace=True)
        smiles_selected = selectable_df.iloc[:batch_size]['SMILES'].values
    else:
        smiles_selected = self.smiles_pool
    return smiles_selected