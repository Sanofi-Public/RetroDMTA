def ratio(self, batch_size):
    if len(self.pool_df) > batch_size:
        pred_score = self.pool_df[f'{self.scoring}_predicted_score']
        uncertainty_score = self.pool_df[f'{self.scoring}_uncertainty_score']
        ratio = uncertainty_score / (self.ratio__epsilon + pred_score)
        selected_smiles = self.pool_df.loc[ratio.sort_values(ascending=False).index[:batch_size], 'SMILES']
    else:
        selected_smiles = self.pool_df['SMILES'].values
    return selected_smiles