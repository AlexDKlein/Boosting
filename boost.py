import numpy as np

class GradientBoost():
    def __init__(self, estimator, n_estimators, loss, learning_rate, subsample=1.0, **kwargs):
        self.estimator = estimator
        self.estimators = []
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.loss = loss
        self.learning_rate = learning_rate
        self.kwargs = kwargs
    
    def fit(self, X, y):
        self._base_pred = pred = np.mean(y)
        self._fit_one_estimator(X, y, pred)
        return self
    
    def predict(self, X):
        return self.learning_rate * sum(est.predict(X) for est in self.estimators) + self._base_pred
    
    def _fit_one_estimator(self, X, y, pred):
        est = self.estimator(**self.kwargs)
        r = self.loss.grad(y, pred)
        est.fit(X, r)
        self.estimators.append(est)
        if len(self.estimators) < self.n_estimators:
            pred += self.learning_rate * est.predict(X)
            self._fit_one_estimator(X, y, pred)
                     
class Loss:
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)
    
    @staticmethod
    def apply(*args, **kwargs):
        raise NotImplementedError
        
    @staticmethod
    def grad(*args, **kwargs):
        raise NotImplementedError
    
class MSE(Loss):
    @staticmethod
    def apply(a, b):
        return np.mean((a - b)**2)
    
    @staticmethod
    def grad(a, b):
        return (a - b) 
    
class LogLoss(Loss):
    @staticmethod
    def apply(a, b):
        return np.mean((a - b)**2)
    
    @staticmethod
    def grad(a, b):
        return 2 * (b - a) / len(a)
    
class ExpLoss(Loss):
    pass