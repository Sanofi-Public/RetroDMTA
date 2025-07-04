# Utils
import os
from dateutil.relativedelta import relativedelta

# Data
import numpy as np
import pandas as pd

# Cheminformatics
from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint, AtomPairFingerprint, MHFPFingerprint

# ML
from models import RandomForestRegressorWithUncertainty, LGBMRegressorWithUncertainty, XGBRegressorWithUncertainty, TanimotoGPRegressor

# Custom
from utils import sum_values
from utility_functions import sigmoid_HTB, sigmoid_LTB, sigmoid_INT
from fingerprints import get_fingerprint_dataframe
from acquisition_functions import get_strategies_dict
from acquisition_functions import *

class Experiment:

    def __init__(self, 
                 experiment_id: str, 
                 dataset: str,
                 initial_date: str,
                 final_date: str,
                 data_path: str, 
                 blueprint_path: str, 
                 log_path: str,
                 timestep: int = 1, 
                 training_threshold: int = 100, 
                 scoring: str = 'geometric',
                 utility_quantile: float = 1.0,
                 model_name: str = 'rf',
                 uncertainty_strategy: str = 'std',
                 n_bootstrap: int = 100,
                 fingerprints = [
                     ECFPFingerprint(fp_size=2048, radius=2, include_chirality=True, count=True, n_jobs=-1),
                     MACCSFingerprint(count=True, n_jobs=-1)
                     ],
                 greediverse__lambda: float = 0.5,
                 greediverse__threshold: float = 0.65,
                 ratio__epsilon : float = 0.95,
                 desired_diversity__threshold: float = 0.7
                 ):
        
        valid_models = ['rf', 'xgb', 'lgbm', 'gp']
        models_uncertainty = {
            'rf': ['std', 'bootstrap'],
            'lgbm': ['quantile', 'bootstrap'],
            'xgb': ['quantile', 'bootstrap'],
            'gp': ['variance']
        }

        if model_name not in valid_models:
            raise ValueError(f"Invalid model '{model_name}'. Valid options are: {valid_models}")

        if uncertainty_strategy not in models_uncertainty[model_name]:
            raise ValueError(
                f"Invalid method '{uncertainty_strategy}' for model '{model_name}'. "
                f"Valid options are: {models_uncertainty[model_name]}"
            )
        

        self.experiment_id = experiment_id
        self.dataset = dataset
        self.iteration = 0

        self.initial_date = pd.to_datetime(initial_date)
        self.final_date = pd.to_datetime(final_date)
        self.current_date = self.initial_date
        self.timestep = timestep

        self.strategies_dict = get_strategies_dict()
        self.scoring = scoring
        self.utility_quantile = utility_quantile
        self.utility_functions = {
            'H': sigmoid_HTB,
            'L': sigmoid_LTB,
            'V': sigmoid_INT
            }
    
        self.training_threshold = training_threshold
        self.model_name = model_name
        self.uncertainty_strategy = uncertainty_strategy
        self.n_bootstrap =  n_bootstrap
        self.fingerprints = fingerprints

        self.blueprint_path = blueprint_path
        self.blueprint = pd.read_csv(self.blueprint_path)
        self.properties = self.blueprint['PROPERTIES'].unique()
        self.weight_dict = dict(zip(self.blueprint['PROPERTIES'], self.blueprint['WEIGHT']))

        self.data_path = data_path
        self.initial_data = pd.read_csv(self.data_path)
        self.data = self.initial_data[['DATE', 'SMILES', 'CPD_ID', 'SERIES_ID'] + list(self.properties)].copy()
        self.data['DATE'] = pd.to_datetime(self.data['DATE'])

        tmp_df = self.data.copy()
        tmp_df = tmp_df[['SMILES', 'DATE']].groupby(by='SMILES', as_index=False).min()

        iterations = 0
        tmp_df['iteration'] = 0
        current_date = self.initial_date
        while current_date < self.final_date:
            iterations += 1
            next_date = current_date + relativedelta(months=timestep)
            tmp_df.loc[(tmp_df['DATE'] >= current_date) & (tmp_df['DATE'] < next_date), 'iteration'] = iterations
            current_date = next_date

        self.batchsize_per_iteration = tmp_df['iteration'].value_counts().sort_index().to_dict()

        self.smiles_all = self.data['SMILES'].unique()
        self.fingerprint_df = get_fingerprint_dataframe(self.smiles_all, self.fingerprints)
        self.mhfp_df = get_fingerprint_dataframe(self.smiles_all, fingerprints=[MHFPFingerprint(fp_size=2048, radius=3, isomeric_smiles=True, variant='bit')])
        self.similarity_matrix = pd.read_parquet(f'../data/{dataset}/similarity_matrix.parquet')

        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)

        self.starting_df = self.data[self.data['DATE'] < initial_date]
        self.smiles_start = self.starting_df['SMILES'].unique()

        self.train_df = self.data[self.data['DATE'] < self.initial_date].drop(columns='DATE').groupby(by=['SMILES', 'CPD_ID', 'SERIES_ID']).mean().reset_index()
        self.train_df['strategy'] = 'initial'
        self.train_df['iteration'] = self.iteration 
        # self.train_df.to_csv(os.path.join(self.iteration_path, 'train_df.csv'), index=False)
        self.train_counts = self.train_df.drop(columns=['SMILES', 'CPD_ID', 'SERIES_ID', 'iteration', 'strategy']).count()

        if self.train_counts.max() < self.training_threshold:
            print(f'⚠️  {self.experiment_id} No property has enough data to start the selection process. Training threshold is set to {self.training_threshold} but the most documented property has {self.train_counts.max()} data points.')

        self.smiles_pool = np.array([])

        # Acquisition functions parameters
        self.greediverse__lambda = greediverse__lambda
        self.greediverse__threshold = greediverse__threshold
        self.ratio__epsilon = ratio__epsilon
        self.desired_diversity__threshold = desired_diversity__threshold
        self.batch_sizes = {}

    def get_next_data(self):

        # Actualize train smiles list
        self.smiles_train = self.train_df['SMILES'].unique()

        # Define next date
        self.next_date = self.current_date + relativedelta(months=self.timestep)

        # Filter to get the next available smiles
        self.smiles_next = self.data[(~self.data['SMILES'].isin(self.smiles_train)) & (self.data['DATE'] < self.next_date)]['SMILES'].unique()

        # Aggregate previous pool with next smiles to create new pool
        self.smiles_pool = np.unique(np.hstack([self.smiles_pool, self.smiles_next]))

        if len(self.smiles_pool) == 0:
            print(f'⚠️ No molecules in the pool set at iteration {self.iteration}.')
        
        strategy_dict, iteration_dict = {}, {}

        for smiles in self.smiles_train:
            strategy_dict[smiles] = self.train_df[self.train_df['SMILES'] == smiles]['strategy'].unique()[0]
            iteration_dict[smiles] = self.train_df[self.train_df['SMILES'] == smiles]['iteration'].unique()[0]

        # Update train_df info
        self.train_df = self.data[(self.data['SMILES'].isin(self.smiles_train)) & (self.data['DATE'] < self.current_date)].drop(columns='DATE').groupby(by=['SMILES', 'CPD_ID', 'SERIES_ID']).mean().reset_index()
        self.train_df['strategy'] = self.train_df['SMILES'].apply(lambda name: strategy_dict.get(name, 'error'))
        self.train_df['iteration'] = self.train_df['SMILES'].apply(lambda name: iteration_dict.get(name, 'error'))
    
        # Increase iteration counter and date
        self.iteration += 1
        self.current_date = self.next_date

        # Save the train dataframe
        self.iteration_path = os.path.join(self.log_path, str(self.iteration))
        os.makedirs(self.iteration_path, exist_ok=True)
        self.train_df.to_csv(os.path.join(self.iteration_path, 'train_df.csv'), index=False)

        # Extract the good molecules in train based on the experimental values
        experimental_df = self.train_df.copy()
        experimental_df['documentation'] = experimental_df[self.properties].notna().sum(axis=1)/len(self.properties)
        experimental_df['total_weight'] = experimental_df[self.properties].apply(lambda row: sum_values(row, self.weight_dict), axis=1)

        for prop in self.properties:
            trend, ltv, htv, weight = np.squeeze(self.blueprint[self.blueprint['PROPERTIES'] == prop][['TREND', 'LTV', 'HTV', 'WEIGHT']].values)
            utility_function = self.utility_functions[trend]
            experimental_df[f'normalized_{prop}'] = experimental_df[prop].apply(lambda x: utility_function(x, ltv, htv, q=self.utility_quantile))

            if self.scoring == 'geometric':
                experimental_df[f"geometric_{prop}"] = experimental_df[f'normalized_{prop}']**weight
                experimental_df['exp_score'] = experimental_df.filter(regex='^geometric_').prod(axis=1)**(1/experimental_df['total_weight'].values)
            elif self.scoring == 'arithmetic':
                experimental_df[f"arithmetic_{prop}"] = experimental_df[f'normalized_{prop}']*weight
                experimental_df['exp_score'] = experimental_df.filter(regex='^arithmetic_').sum(axis=1)/experimental_df['total_weight'].values

        self.good_in_train = experimental_df[(experimental_df['exp_score'] > 0.5) & (experimental_df['documentation'] > 0.65)]['SMILES'].values


    def setup_training(self):

        # Define a dictionnary that will contain the X_train and y_train for each property
        self.train_dict = {}
        self.valid_models = []
        for prop in self.properties:
            # Filter the train_df for the defined property
            tmp_df = self.train_df[['SMILES', prop]].dropna()
            trend, ltv, htv, weight = np.squeeze(self.blueprint[self.blueprint['PROPERTIES'] == prop][['TREND', 'LTV', 'HTV', 'WEIGHT']].values)
            if len(tmp_df) >= self.training_threshold:
                for index, row in tmp_df.iterrows():
                        if trend == 'H':
                            if row[prop] >= ltv:
                                self.valid_models.append(prop)
                                tmp_dict = {}
                                tmp_dict['X_train'] = np.vstack(tmp_df.merge(self.fingerprint_df[['SMILES', 'fingerprint']], on='SMILES')['fingerprint'].values)
                                tmp_dict['y_train'] = tmp_df[prop].values
                                self.train_dict[prop] = tmp_dict
                                break
                        elif trend == 'L':
                            if row[prop] <= htv:
                                self.valid_models.append(prop)
                                tmp_dict = {}
                                tmp_dict['X_train'] = np.vstack(tmp_df.merge(self.fingerprint_df[['SMILES', 'fingerprint']], on='SMILES')['fingerprint'].values)
                                tmp_dict['y_train'] = tmp_df[prop].values
                                self.train_dict[prop] = tmp_dict
                                break
                        elif trend == 'V':
                            if ltv <= row[prop] <= htv:
                                self.valid_models.append(prop)
                                tmp_dict = {}
                                tmp_dict['X_train'] = np.vstack(tmp_df.merge(self.fingerprint_df[['SMILES', 'fingerprint']], on='SMILES')['fingerprint'].values)
                                tmp_dict['y_train'] = tmp_df[prop].values
                                self.train_dict[prop] = tmp_dict
                                break

    def start_training(self):
        
        # Define a dictionnary that will contain the trained models
        self.models_dict = {}

        for prop in list(self.train_dict.keys()):
            X_train = self.train_dict[prop]['X_train']
            y_train = self.train_dict[prop]['y_train']

            if self.model_name == 'rf':
                estimator = RandomForestRegressorWithUncertainty(uncertainty_method=self.uncertainty_strategy, n_bootstrap_samples=self.n_bootstrap)
            elif self.model_name == 'lgbm':
                estimator = LGBMRegressorWithUncertainty(uncertainty_method=self.uncertainty_strategy, n_bootstrap_samples=self.n_bootstrap)
            elif self.model_name == 'xgb':
                estimator = XGBRegressorWithUncertainty(uncertainty_method=self.uncertainty_strategy, n_bootstrap_samples=self.n_bootstrap)
            elif self.model_name == 'gp':
                estimator = TanimotoGPRegressor(uncertainty_method=self.uncertainty_strategy)

            estimator.fit(X_train, y_train)
            self.models_dict[prop] = estimator

            # Add function to save models if needed

    def process_pool(self):

        self.pool_df = pd.DataFrame()
        if len(self.smiles_pool) != 0:

            self.pool_df['SMILES'] = self.smiles_pool
            X_pool = np.vstack(self.pool_df.merge(self.fingerprint_df[['SMILES', 'fingerprint']], on='SMILES')['fingerprint'].values)

            columns_to_add_first = {}

            for prop in list(self.train_dict.keys()):
                estimator = self.models_dict[prop]
                predictions, uncertainties = estimator.predict_with_uncertainty(X_pool)
                min_uncertainty, max_uncertainty = np.min(uncertainties), np.max(uncertainties)

                if min_uncertainty == max_uncertainty:
                    normalized_uncertainty = np.nan
                else:
                    normalized_uncertainty = ((uncertainties - min_uncertainty) /
                                            (max_uncertainty - min_uncertainty))
                    
                    normalized_uncertainty = np.where(normalized_uncertainty < 0.05, 0.05, normalized_uncertainty)

                # Store computed columns in a dictionary
                columns_to_add_first[f'uncertainty_{prop}'] = uncertainties
                columns_to_add_first[f'normalized_uncertainty_{prop}'] = normalized_uncertainty

                # Extract trend, ltv, htv
                trend, ltv, htv = np.squeeze(
                    self.blueprint[self.blueprint['PROPERTIES'] == prop][['TREND', 'LTV', 'HTV']].values
                )
                sigmoid_func = self.utility_functions[trend]

                # Store predictions
                columns_to_add_first[f'predicted_{prop}'] = predictions
                columns_to_add_first[f'normalized_predicted_{prop}'] = np.array([
                    sigmoid_func(x=value, LTV=ltv, HTV=htv, q=self.utility_quantile)
                    for value in predictions
                ])

            # Add all columns from the first loop in one operation
            temp_df_first = pd.DataFrame(columns_to_add_first, index=self.pool_df.index)
            self.pool_df = pd.concat([self.pool_df, temp_df_first], axis=1)

            columns_to_add_second = {}
            total_weight = 0

            for prop in list(self.train_dict.keys()):
                property_weight = self.weight_dict[prop]

                norm_unc_col = self.pool_df[f'normalized_uncertainty_{prop}'].values
                norm_pred_col = self.pool_df[f'normalized_predicted_{prop}'].values

                # Geometric and arithmetic uncertainty
                columns_to_add_second[f"geometric_uncertainty_{prop}"] = norm_unc_col ** property_weight
                columns_to_add_second[f"arithmetic_uncertainty_{prop}"] = norm_unc_col * property_weight

                # Geometric and arithmetic predictions
                columns_to_add_second[f"geometric_predicted_{prop}"] = norm_pred_col ** property_weight
                columns_to_add_second[f"arithmetic_predicted_{prop}"] = norm_pred_col * property_weight

                total_weight += property_weight

            # Add all columns from the second loop in one operation
            temp_df_second = pd.DataFrame(columns_to_add_second, index=self.pool_df.index)
            self.pool_df = pd.concat([self.pool_df, temp_df_second], axis=1)

            self.pool_df = self.pool_df.copy() # avoid fragmentation error in pandas
            self.pool_df['geometric_uncertainty_score'] = self.pool_df.filter(regex='^geometric_uncertainty_').prod(axis=1)**(1/total_weight)
            self.pool_df['arithmetic_uncertainty_score'] = self.pool_df.filter(regex='^arithmetic_uncertainty_').sum(axis=1)/total_weight

            self.pool_df['geometric_predicted_score'] = self.pool_df.filter(regex='^geometric_predicted_').prod(axis=1)**(1/total_weight)
            self.pool_df['arithmetic_predicted_score'] = self.pool_df.filter(regex='^arithmetic_predicted_').sum(axis=1)/total_weight

        self.pool_df.to_csv(os.path.join(self.iteration_path, 'pool_df.csv'), index=False)

    
    def select_molecules(self, selection_strategies, batch_sizes):

        self.smiles_selected = np.array([])
        self.selected_pool_df = pd.DataFrame()
        self.selected_train_df = pd.DataFrame()

        if len(self.smiles_pool) != 0:

            for selection_strategy, batch_size in zip(selection_strategies, batch_sizes):
                
                if batch_size != 0:
                    
                    # Define selection strategy 
                    strategy = self.strategies_dict[selection_strategy]

                    # Select molecules 
                    strategy_smiles_selected = strategy(self, batch_size=batch_size)
                    self.smiles_selected = np.hstack([self.smiles_selected, strategy_smiles_selected])

                    #
                    strategy_selected_pool_df = self.pool_df[self.pool_df['SMILES'].isin(strategy_smiles_selected)].copy()
                    strategy_selected_pool_df['strategy'] = selection_strategy
                    strategy_selected_pool_df['iteration'] = self.iteration

                    self.selected_pool_df = pd.concat([self.selected_pool_df, strategy_selected_pool_df])

                    #
                    self.smiles_pool = np.setdiff1d(self.smiles_pool, strategy_smiles_selected)
                    self.pool_df = self.pool_df[~self.pool_df['SMILES'].isin(strategy_smiles_selected)]

                    # 
                    strategy_selected_train_df = self.data[(self.data['SMILES'].isin(strategy_smiles_selected)) & (self.data['DATE'] < self.current_date)].drop(columns='DATE').groupby(by=['SMILES', 'CPD_ID', 'SERIES_ID']).mean().reset_index().copy()
                    strategy_selected_train_df['strategy'] = selection_strategy
                    strategy_selected_train_df['iteration'] = self.iteration

                    self.selected_train_df = pd.concat([self.selected_train_df, strategy_selected_train_df])
                
            self.train_df = pd.concat([self.train_df, self.selected_train_df])
            self.batch_sizes[self.iteration] = batch_sizes

        self.selected_pool_df.to_csv(os.path.join(self.iteration_path, 'selected_pool_df.csv'), index=False)
        self.selected_train_df.to_csv(os.path.join(self.iteration_path, 'selected_train_df.csv'), index=False) 