import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class MultiLabelXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.models = []
        self.default_params = {}

    def set_params(self, **params):
        self.default_params = params


    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        predictions = []

        for i in range(y.shape[1]):
            model = xgb.XGBClassifier()
            y_ = y[:, i]
            params = self.default_params.copy()
            pos_weight = (y_.shape[0]-y_.sum())/y_.sum()
            params['scale_pos_weight'] = pos_weight
            model.set_params(**params)
            X_ = np.hstack((X, *predictions))
            model.fit(X_, y_)
            predictions.append(model.predict(X_).reshape(-1,1))
            self.models.append(model)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self


    def predict_proba(self, X):
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        pred = [ ]
        for m in self.models:
            y_pred = m.predict_proba(X)[:, 1]
            pred.append(y_pred.reshape(-1,1))
        pred = np.hstack(pred)
        return pred

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        predictions = []
        pred = [ ]
        for m in self.models:
            y_pred = m.predict(np.hstack((X, *pred)))
            # y_pred = m.predict(X)
            pred.append(y_pred.reshape(-1,1))
        pred = np.hstack(pred)
        return pred