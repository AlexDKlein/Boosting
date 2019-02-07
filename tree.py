import numpy as np

class Builder():
    def __init__(self, splitter, 
                 max_depth = None, 
                 min_leaf_size = 1,
                 classifier=False):
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.labels = []
        self.probas = []
        self.indices = []
        self.values = []
        self.left = []
        self.right = []
        self.classes = None
        self.classifier = classifier
        self.n_nodes = 0
        self.depth = 0

    def fit(self, X, y):
        if self.classifier:
            self.classes = np.unique(y)
        self.branch(X, y)
        return self
        
    def branch(self, X, y, depth=0):
        if self.classifier:
            self.classes = np.unique(y)
        node, mask  = self.create_node(X, y)
        if mask is None: 
            return node
        left_X, left_y = X[mask], y[mask] 
        right_X, right_y =  X[~mask], y[~mask] 
        if all([
            self.max_depth is None or depth < self.max_depth,
            min([len(left_y), len(right_y)]) >= self.min_leaf_size
        ]): 
            self.left[node]  = self.branch(left_X,  left_y,  depth + 1)
            self.right[node] = self.branch(right_X, right_y, depth + 1)
        self.depth = max(self.depth, depth)
        return node
        
    def create_empty(self):
        self.left.append(-1)
        self.right.append(-1)
        self.labels.append(None)
        self.probas.append([])
        self.indices.append(None)
        self.values.append(None)
        self.n_nodes += 1
        
    def create_node(self, X, y):
        self.create_empty()
        #Only needed if leaf; could be moved to Tree class
        if self.classifier:
            cnts = (y == self.classes[:, None]).sum(axis=-1)
            self.labels[-1] = self.classes[cnts.argmax()]
            self.probas[-1] = cnts / len(y)
        else:
            self.labels[-1] = y.mean()
        
        idx, val, mask = self.splitter._choose_index(X, y)
        self.indices[-1] = idx
        self.values[-1] = val
        return self.n_nodes - 1, mask
    
class Splitter():
    def __init__(self, criterion):
        self.criterion = criterion
    
    def _choose_index(self, X, y):
        if len(X) != y.size:
            raise ValueError(f'X of shape {X.shape} and y of shape {y.shape} do not align.')
        idx, val, mask = None, None, None
        gain = 0
        for i, col in enumerate(X.T):
            values = np.unique(col)
            for v in values:
                m = self.mask(col, v)
                g = self._information_gain(y, m)
                if g > gain: 
                    idx, val, gain, mask = i,v,g,m
        return idx, val, mask
    
    def _information_gain(self, y, mask):
        impurity = self.criterion(y)
        y1, y2 = y[mask], y[~mask]
        split_impurity = (y1.size * self.criterion(y1) 
                        + y2.size * self.criterion(y2)) / y.size
        return impurity - split_impurity
   
    @staticmethod
    def split(X, y, index=0, value=0, discrete=False):
        mask = Splitter.mask(X[:, index], value, discrete)
        return y[mask], y[~mask]
    
    @staticmethod
    def mask(X, value, discrete=False):
        return X == value if discrete else X >= value
    
class Tree():
    def __init__(self, metric='entropy', max_depth=None, min_leaf_size=1, classifier=True):
        self.classifier = classifier
        self.categorical = None
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.n_nodes = 0
        self.indices = None
        self.values = None
        self.labels = None
        self.probas = None
        self.left = None
        self.right = None
        self.leaf = None
        self.n_features = None
        self.splitter = Splitter(getattr(self, f'_{metric}'))
    
    def fit(self, X, y):
        self.n_features = X.size // y.size
        
        y = y.ravel()
        X = X.reshape(y.size, self.n_features)
        
        self.categorical = np.vectorize(lambda x: isinstance(x, (str, bool)))(X[0])
        builder = Builder(self.splitter, self.max_depth, self.min_leaf_size, self.classifier)
        builder.fit(X, y)
        self.indices = np.array(builder.indices)
        self.values = np.array(builder.values)
        self.labels = np.array(builder.labels)
        self.probas = np.array(builder.probas)
        self.left = np.array(builder.left)
        self.right = np.array(builder.right)
        self.n_nodes = builder.n_nodes
        self.depth = builder.depth
        self.leaf = (self.left == self.right)
        return self
        
    def predict_path(self, X, node=0):
        X = X.reshape(-1, self.n_features)
        if self.leaf[node]:
            return node
        idxs = self.indices[node]
        vals = self.values[node]
        mask = self.splitter.mask(X[:, idxs], vals, discrete=self.categorical[idxs])
        output = np.empty(mask.size, dtype=int)
        output[mask]  = self.predict_path(X[mask],  self.left[node])
        output[~mask] = self.predict_path(X[~mask], self.right[node])
        return output
    
    def predict(self, X):
        final_node = self.predict_path(X)
        return self.labels[final_node]

    @staticmethod
    def _entropy(y):
        y = y.ravel()
        p = np.bincount(y) / y.size
        return - (p * np.log(p, where=p.astype(bool))).sum()
    
    @staticmethod
    def _gini(y):
        y = y.ravel()
        p = np.bincount(y) / y.size
        return 1 - (p**2).sum()
    
    @staticmethod
    def _var(y):
        return np.var(y)