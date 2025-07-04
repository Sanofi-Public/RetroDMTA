def uncertainty(self, batch_size):
    if len(self.pool_df) > batch_size:
        selected_smiles = self.pool_df.sort_values(by=f'{self.scoring}_uncertainty_score', ascending=False)['SMILES'].values[:batch_size]
    else:
        selected_smiles = self.pool_df['SMILES'].values
    return selected_smiles