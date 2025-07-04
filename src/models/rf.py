from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
import numpy as np

class RandomForestRegressorWithUncertainty:
    def __init__(self, uncertainty_method='std', n_bootstrap_samples=50, **kwargs):

        self.uncertainty_method = uncertainty_method
        self.n_bootstrap_samples = n_bootstrap_samples
        self.bootstrap_models_ = []
        self.params = kwargs

    def fit(self, X, y):

        if self.uncertainty_method == 'std':
            model = RandomForestRegressor(n_jobs=-1, random_state=42, **self.params)
            model.fit(X, y)
            self.model = model
            self.estimators_ = model.estimators_

        elif self.uncertainty_method == 'bootstrap':
            for _ in range(self.n_bootstrap_samples):
                X_resampled, y_resampled = resample(X, y)
                model = RandomForestRegressor(n_jobs=-1, random_state=42, **self.params)
                model.fit(X_resampled, y_resampled)
                self.bootstrap_models_.append(model)


    def predict_with_uncertainty(self, X):

        if self.uncertainty_method == 'std':
            predictions = self.model.predict(X)
            all_preds = np.array([tree.predict(X) for tree in self.estimators_])
            uncertainties = np.std(all_preds, axis=0)

        elif self.uncertainty_method == 'bootstrap':
            preds = np.array([model.predict(X) for model in self.bootstrap_models_])
            predictions = preds.mean(axis=0)
            uncertainties = preds.std(axis=0)

        return predictions, uncertainties