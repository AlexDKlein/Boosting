import numpy as np

class Bootstrapper():
    def __init__(self, base_estimator, n_estimators, **kwargs):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = None
        self.kwargs = kwargs

    def fit(self, X, y=None):
        self._bootstrap(X, y)

    def predict(self, X):
        return np.mean([estimator.predict(X) for estimator in self.estimators], axis=0)

    def _bootstrap(self, X, y=None):
        self.estimators = []
        for _ in range(self.n_estimators):
            estimator = self.base_estimator(**self.kwargs)
            bag_idxs = self._bag_indices(len(X))
            y_ = y[bag_idxs] if y is not None else y
            X_ = X[bag_idxs]
            estimator.fit(X_, y_)
            self.estimators.append(estimator)
    
    def _bag_indices(self, n):
        idxs = np.random.randint(0, n, n)
        return idxs
