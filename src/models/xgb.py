from xgboost import XGBRegressor
from sklearn.utils import resample
import numpy as np

class XGBRegressorWithUncertainty:

    def __init__(self, uncertainty_method='bootstrap', n_bootstrap_samples=50,
                 quantiles=(0.025, 0.975), **kwargs):
        
        self.uncertainty_method = uncertainty_method
        self.n_bootstrap_samples = n_bootstrap_samples
        self.quantiles = quantiles
        self.bootstrap_models_ = []
        self.models_ = {}
        self.params = kwargs

    def fit(self, X, y):

        if self.uncertainty_method == 'quantile':

            for q in [0.5] + list(self.quantiles):
                model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=q, n_jobs=-1, **self.params)
                model.fit(X, y)
                self.models_[q] = model

        elif self.uncertainty_method == 'bootstrap':
            for _ in range(self.n_bootstrap_samples):
                X_resampled, y_resampled = resample(X, y)
                model = XGBRegressor(n_jobs=-1, **self.params)
                model.fit(X_resampled, y_resampled)
                self.bootstrap_models_.append(model)

    def predict_with_uncertainty(self, X):

        if self.uncertainty_method == 'quantile':
            lower_q, upper_q = self.quantiles
            lower_preds = self.models_[lower_q].predict(X)
            median_preds = self.models_[0.5].predict(X)
            upper_preds = self.models_[upper_q].predict(X)
            predictions = median_preds
            uncertainties = (upper_preds - lower_preds) / 2

        elif self.uncertainty_method == 'bootstrap':
            preds = np.array([model.predict(X) for model in self.bootstrap_models_])
            predictions = preds.mean(axis=0)
            uncertainties = preds.std(axis=0)

        return predictions, uncertainties
