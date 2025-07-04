import numpy as np
import pandas as pd
from utils import sum_values

def oracle(self, batch_size):

    if len(self.smiles_pool) > batch_size:

        oracle_df = self.data.copy()

        # Define the aggregation dictionary
        agg_dict = {'DATE': 'min'}

        # Apply 'mean' to all other columns except the grouping column and the column with 'min'
        for col in oracle_df.columns:
            if col not in ['SMILES', 'CPD_ID', 'SERIES_ID', 'DATE']:
                agg_dict[col] = 'mean'

        oracle_df = oracle_df.groupby(['SMILES', 'CPD_ID', 'SERIES_ID']).agg(agg_dict).reset_index()

        oracle_df = oracle_df.copy()

        weight_dict = dict(zip(self.blueprint['PROPERTIES'], self.blueprint['WEIGHT']))
        props = sorted(self.blueprint['PROPERTIES'].unique())

        for prop in props:
            trend, ltv, htv, weight = np.squeeze(self.blueprint[self.blueprint['PROPERTIES'] == prop][['TREND', 'LTV', 'HTV', 'WEIGHT']].values)
            sigmoid_function = self.utility_functions[trend]
            oracle_df[f'normalized_{prop}'] = oracle_df[prop].apply(lambda x: sigmoid_function(x, ltv, htv, q=1))
            oracle_df = oracle_df.copy()
            oracle_df[f"geometric_{prop}"] = oracle_df[f'normalized_{prop}']**weight
            oracle_df = oracle_df.copy()
            oracle_df[f"arithmetic_{prop}"] = oracle_df[f'normalized_{prop}']*weight

        oracle_df = oracle_df.copy()
        oracle_df['total_weight'] = oracle_df[props].apply(lambda row: sum_values(row, weight_dict), axis=1)
        oracle_df['exp_geometric_score'] = oracle_df.filter(regex='^geometric_').prod(axis=1)**(1/oracle_df['total_weight'].values)
        oracle_df['exp_arithmetic_score'] = oracle_df.filter(regex='^arithmetic_').sum(axis=1)/oracle_df['total_weight'].values
        oracle_df['documentation'] = oracle_df[props].notna().sum(axis=1)/len(props)
        oracle_df = oracle_df[['SMILES', 'exp_geometric_score', 'documentation']].copy()
        oracle_df['oracle_score'] = oracle_df['exp_geometric_score'] * oracle_df['documentation']
        oracle_df = oracle_df[oracle_df['SMILES'].isin(self.smiles_pool)]
        smiles_selected = oracle_df[oracle_df['SMILES'].isin(self.smiles_pool)].sort_values(by=['oracle_score', 'documentation'], ascending=[False, True])['SMILES'].values[:batch_size]
    else:
        smiles_selected = self.smiles_pool
    return smiles_selected
