import numpy as np
import pandas as pd

def desired_diversity(self, batch_size):

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

            new_scores = []
            for index, row in tmp_df.iterrows():
                smiles = row['SMILES']
                score = row['score']
                max_similarity = self.similarity_matrix.loc[smiles, smiles_train + smiles_selected].max()

                if max_similarity > self.desired_diversity__threshold:
                    alpha = 0
                else:
                    alpha = 1
                
                new_score = score * alpha
                new_scores.append(new_score)
                
            tmp_df['new_score'] = np.vstack(new_scores)

            if len(tmp_df[tmp_df['new_score'] > 0]) > 0:
                tmp_df = tmp_df[tmp_df['new_score'] > 0].copy()

            selected_molecule = tmp_df.loc[tmp_df['new_score'].idxmax()]['SMILES']
            smiles_selected.append(selected_molecule)
    else:
        smiles_selected = self.smiles_pool
    return smiles_selected












# import numpy as np
# import pandas as pd

# def desired_diversity(self, batch_size):

#     if len(self.smiles_pool) > batch_size:
#         smiles_pool = self.smiles_pool
#         pool_scores = self.pool_df[f'{self.scoring}_predicted_score'].values
#         selectable_df = pd.DataFrame()
#         selectable_df['SMILES'] = smiles_pool
#         selectable_df['score'] = pool_scores

#         smiles_train = list(self.smiles_train)

#         smiles_selected = []

#         for i in range(batch_size):

#             tmp_df = selectable_df.copy()
#             tmp_df = tmp_df[~tmp_df['SMILES'].isin(smiles_selected)]

#             new_scores = []
#             for index, row in tmp_df.iterrows():
#                 smiles = row['SMILES']
#                 score = row['score']
#                 alpha = 1
#                 for smi in smiles_selected + smiles_train:
#                     similarity = self.similarity_matrix.loc[smi, smiles]
#                     if similarity > self.desired_diversity__threshold:
#                         alpha = 0
#                         break
                
#                 new_score = score * alpha

#                 new_scores.append(new_score)
                
#             tmp_df['new_score'] = np.vstack(new_scores)

#             if len(tmp_df[tmp_df['new_score'] > 0]) > 0:
#                 tmp_df = tmp_df[tmp_df['new_score'] > 0].copy()

#             selected_molecule = tmp_df.loc[tmp_df['new_score'].idxmax()]['SMILES']
#             smiles_selected.append(selected_molecule)
#     else:
#         smiles_selected = self.smiles_pool
#     return smiles_selected

