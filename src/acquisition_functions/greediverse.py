import numpy as np
import pandas as pd

# def greediverse(self, batch_size):

#     if len(self.smiles_pool) > batch_size:
#         smiles_pool = self.smiles_pool
#         scores = self.pool_df[f'{self.scoring}_predicted_score'].values
#         stds = np.sqrt(scores * (1 - scores))
#         greediverse_df = pd.DataFrame()
#         greediverse_df['SMILES'] = smiles_pool
#         greediverse_df['score'] = scores
#         greediverse_df['std'] = stds

#         smiles_selected = []

#         for i in range(batch_size):

#             tmp_df = greediverse_df.copy()
#             tmp_df = tmp_df[~tmp_df['SMILES'].isin(smiles_selected)]

#             new_scores = []
#             for index, row in tmp_df.iterrows():
#                 smiles = row['SMILES']
#                 score = row['score']
#                 std = row['std']
#                 penalty = 0
#                 for smi in smiles_selected:
#                     similarity = self.similarity_matrix.loc[smi, smiles]
#                     if similarity < self.greediverse__threshold:
#                         similarity = 0
#                     std_smi = greediverse_df[greediverse_df['SMILES'] == smi]['std'].values
#                     penalty += std_smi * similarity
                
#                 new_score = score - self.greediverse__lambda * std * penalty

#                 new_scores.append(new_score)

#             tmp_df['new_score'] = np.vstack(new_scores)

#             selected_molecule = tmp_df.loc[tmp_df['new_score'].idxmax()]['SMILES']
#             smiles_selected.append(selected_molecule)
#     else:
#         smiles_selected = self.smiles_pool
#     return smiles_selected

def greediverse(self, batch_size):
    if len(self.smiles_pool) <= batch_size:
        return self.smiles_pool

    # Precompute scores and std values from the pool
    smiles_pool = np.array(self.smiles_pool)
    scores = self.pool_df[f'{self.scoring}_predicted_score'].values
    stds = np.sqrt(scores * (1 - scores))
    
    # Create a DataFrame for convenience (we will mostly use numpy arrays)
    greediverse_df = pd.DataFrame({
        'SMILES': smiles_pool,
        'score': scores,
        'std': stds
    })
    
    # Precompute a dictionary for std values for each molecule
    std_dict = dict(zip(greediverse_df['SMILES'], greediverse_df['std']))
    
    selected_smiles = []

    for _ in range(batch_size):
        # Filter out already selected SMILES using numpy's isin for speed.
        mask = ~np.isin(greediverse_df['SMILES'].values, selected_smiles)
        candidate_df = greediverse_df[mask].reset_index(drop=True)
        candidate_smiles = candidate_df['SMILES'].values
        candidate_scores = candidate_df['score'].values
        candidate_stds = candidate_df['std'].values

        # Initialize the penalty as zero for each candidate.
        penalty = np.zeros(candidate_scores.shape[0], dtype=float)

        # If we have any selected molecules, add their penalty contributions.
        for sel in selected_smiles:
            # Grab the similarity vector for the selected molecule with all candidate molecules.
            # Here we assume that self.similarity_matrix is a DataFrame indexed by SMILES
            sim_vec = self.similarity_matrix.loc[sel, candidate_smiles].values.astype(float)
            
            # Zero out similarities below the threshold
            sim_vec[sim_vec < self.greediverse__threshold] = 0
            
            # Accumulate the penalty: multiply by the precomputed std value.
            penalty += std_dict[sel] * sim_vec

        # Compute the adjusted score vector in a vectorized fashion.
        new_scores = candidate_scores - self.greediverse__lambda * candidate_stds * penalty

        # Choose the candidate with the highest new score.
        best_idx = new_scores.argmax()
        best_smile = candidate_smiles[best_idx]
        selected_smiles.append(best_smile)
    
    return selected_smiles
