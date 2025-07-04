import numpy as np

def random(self, batch_size):
    if len(self.pool_df) > batch_size:
        selected_smiles = np.random.choice(self.pool_df['SMILES'].values, batch_size, replace=False)
    else:
        selected_smiles = self.pool_df['SMILES'].values
    return selected_smiles