def harmonic(self, batch_size):
    if len(self.pool_df) > batch_size:
        pred_score = self.pool_df[f'{self.scoring}_predicted_score']
        uncertainty_score = self.pool_df[f'{self.scoring}_uncertainty_score']
        selected_smiles = self.pool_df.loc[((2*pred_score*uncertainty_score)/(pred_score+uncertainty_score)).sort_values(ascending=False).index[:batch_size], 'SMILES']
    else:
        selected_smiles = self.pool_df['SMILES'].values
    return selected_smiles